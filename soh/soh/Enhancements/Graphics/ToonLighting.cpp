// Wind Waker-style toon lighting -- game-side policy.
//
// libultraship owns the per-pixel transport: it relights the draws SoH brackets with gSPToon, using
// a directional key, optional local contributions and a generic ramp (SetToonRamp).
// This module owns the OoT-specific
// policy that the framework must never know about:
//   - sun/moon plus up to four local lights, or the legacy closest-light key,
//   - how the key eases from one source to another (per-actor persistent state),
//   - the look tuning (ramp parameters), pushed once per frame.
// The framework never reads SoH's CVars; everything it needs is pushed from here.

#include <libultraship/bridge.h>
#include <ship/Context.h>
#include <fast/Fast3dWindow.h>
#include <fast/interpreter.h>

#include "soh/Enhancements/game-interactor/GameInteractor.h"
#include "soh/ShipInit.hpp"
#include "soh/cvar_prefixes.h"
#include "soh/ActorDB.h" // SOH [Enhancement] actor-name lookup for the cel-shading blacklist
#include "soh/Enhancements/Graphics/ToonLighting.h"
// Declares the FrameInterpolation_Record* functions (with C linkage) that the OPEN_DISPS/CLOSE_DISPS
// macros call, so their references in this TU match the definitions. Must precede any OPEN_DISPS use.
#include "soh/frame_interpolation.h"

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>

extern "C" {
#include "z64.h"
#include "macros.h"
#include "functions.h"
#include "variables.h"
// For identifying Navi (Link's fairy) so players can opt her out of cel key selection: the EnElf struct +
// FairyType (FAIRY_NAVI). Same approach the light-casting feature uses.
#include "overlays/actors/ovl_En_Elf/z_en_elf.h"
extern PlayState* gPlayState;
}

// ---------------------------------------------------------------------------------------------------
// Ramp parameters (frame-global look tuning)
// ---------------------------------------------------------------------------------------------------

// Defaults match the GUI slider DefaultValue()s, so a fresh install (CVar unset) renders the same as
// the slider's default position. (These intentionally differ from libultraship's neutral
// TOON_SHADING_DEFAULT_* framework fallbacks -- SoH always pushes these every frame.)
static constexpr float kDefaultRampCenter = 0.5f;
static constexpr float kDefaultRampSoftness = 0.02f;
static constexpr float kDefaultHighlightIntensity = 0.6f;
static constexpr float kDefaultShadowIntensity = 0.6f;

// Selection defaults (game-side only -- these never reach the framework).
static constexpr float kDefaultPointLightRange = 1.5f;
static constexpr float kDefaultTransitionTime = 1.0f;

// Actor-shadow defaults (the "Actor Shadows" page; CVar prefix Graphics.WorldShadows.*). Opacity is the
// core blend strength; length maps (inversely) to how far a low-angle key may stretch the shadow (higher =
// longer); slab depth/rise bound the ground band below/above the feet. All are game-side policy pushed once
// per frame (look) or per object. Keep these in sync with the GUI slider DefaultValue()s.
static constexpr float kDefaultShadowOpacity = 0.2f;
static constexpr float kDefaultShadowLength = 0.2f;
static constexpr float kDefaultShadowSlabDepth = 8.0f; // stencil-volume depth below the feet (ground band)
static constexpr float kDefaultShadowSlabRise = 8.0f;  // stencil-volume height above the feet (uphill ground)
static constexpr int kDefaultShadowEdgeSoftness = 0;   // penumbra rings around the silhouette (0 = hard edge)
static constexpr int kDefaultShadowMaxDistance = 550;  // camera-forward distance past which shadows are culled
static constexpr float kShadowFadeTime = 0.15f;        // seconds to ease the shadow size in/out (anti-pop, like Navi)

// Per-frame snapshot of every CVar the per-actor hot path reads. CVarGet* is a string-keyed hash-map
// lookup that heap-allocates for keys this long, and HandleActorDraw runs for EVERY drawn actor every
// frame -- reading them once per frame here removes thousands of lookups (and allocations) per second.
// Refreshed at the top of each game frame (OnToonFrameUpdate) and by RegisterToonLighting, so both menu
// and console changes take effect within a frame.
static struct {
    bool cel = true;
    int shadowMode = SHADOW_MODE_VANILLA;
    bool shadows = false;            // shadowMode == SHADOW_MODE_ACTOR (stencil volumes)
    bool shadowMap = false;          // shadowMode == SHADOW_MODE_SHADOW_MAP (cascaded depth maps)
    bool shadowMapSupported = false; // the running backend implements the depth pass (D3D11 only)
    // Radial distance from the camera past which an actor is not worth capturing as a shadow caster: the
    // furthest active cascade, since beyond that there is no depth map left to record it in.
    f32 shadowMapReach = SHADOW_MAP_DEFAULT_SPLIT_2;
    f32 shadowMapCasterDrawRadius = 0.0f;
    // SOH [Enhancement] Whether shadow-map mode reorders the frame so the actor loop draws BEFORE the room
    // (see z_play.c). It exists to take a frame of lag off a moving character's shadow, and it changes the
    // order actors and the room are submitted in -- which is the order translucent geometry is composited
    // in.
    //
    // OFF by default, after a report it was responsible for: in the Graveyard, Navi drew through the paving
    // stones, and only with shadow maps on. Confirmed by switching this alone.
    //
    // The trade it was making was taken unconditionally and is the wrong way round. What it buys is a frame
    // of freshness on a MOVING CHARACTER'S OWN shadow -- which the block in z_play.c itself calls invisible
    // on scenery, since scenery does not move. What it costs is the submission order of every translucent
    // surface in the frame, which produces faults a player meets in ordinary play.
    //
    // Still switchable: the lag it removes is real, and someone who does not hit the compositing fault may
    // prefer the sharper result.
    bool shadowMapCasterFirst = false;
    // The direction the key light TRAVELS, world space, normalised -- the same vector the cascades are
    // built from, kept here so the caster-reach test can follow a shadow along it. Straight down until the
    // first frame computes it.
    f32 shadowMapLightDir[3] = { 0.0f, -1.0f, 0.0f };
    bool suppressVanilla = true;
    bool useNaviLight = true;
    bool multipleLights = true;
    f32 localIntensity = 0.5f;
    bool showDebug = false;
    // The shadow-map debug view (ShadowMap.ShowCascadeBounds). Non-zero also turns on the scenery-caster
    // census below -- the view says SOMETHING is casting, the census says what.
    bool shadowMapCensus = false;
    f32 pointRange = kDefaultPointLightRange;
    f32 transitionTime = kDefaultTransitionTime;
    f32 maxDist = (f32)kDefaultShadowMaxDistance;
} sParams;

static Fast::GfxRenderingAPI* GetRenderingApi(); // defined with the other Fast3D accessors below
// Key light from the environment directionals (sun or moon, whichever is brighter); defined below.
static void ToonEnvKey(PlayState* play, f32 dirOut[3], f32 colOut[3]);

// ===================================================================================================
// Scenery-caster census.
//
// The debug view (ShadowMap.ShowCascadeBounds = 2) answers "is this being captured, and into which
// layer" and it has settled several arguments that reasoning about the display lists got wrong every
// time. What it cannot answer is WHICH ACTOR a green blob came from, and that is the question every one
// of those arguments actually turned on: the caster layer is chosen by category, the categories sort
// actors by what the engine does with them rather than what they are made of, and so the only reliable
// way to name the thing casting is to have the code that classified it say so.
//
// This counts the actors that opened the scenery bracket, per id, over a short window, and publishes a
// readable table for the menu to display beside the debug slider. Costs an integer bump per scenery
// actor and only while the debug view is on.
// ===================================================================================================
static std::unordered_map<s16, u32> sCensusAccum; // id -> bracket opens accumulated this window
static u32 sCensusFrames = 0;                     // frames accumulated into the above
static std::string sCensusText;                   // published table, read by the menu

static void ToonShadowCensusPublish() {
    if (!sParams.shadowMapCensus) {
        if (!sCensusAccum.empty()) {
            sCensusAccum.clear();
            sCensusFrames = 0;
            sCensusText.clear();
        }
        return;
    }
    sCensusFrames++;
    if (sCensusFrames < 20) { // ~1/3 s: long enough that a flickering count settles, short enough to feel live
        return;
    }

    std::vector<std::pair<s16, u32>> rows(sCensusAccum.begin(), sCensusAccum.end());
    std::sort(rows.begin(), rows.end(), [](const std::pair<s16, u32>& a, const std::pair<s16, u32>& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });

    sCensusText.clear();
    if (rows.empty()) {
        sCensusText = "(nenhum cenário projetando)";
    }
    // Only the busiest handful: a scattered particle is by definition many instances, so whatever is
    // wanted here is at the top. Averaged over the window, because an actor drawn every frame should read
    // as its instance count and not as a multiple of it.
    const size_t kMaxRows = 12;
    for (size_t i = 0; i < rows.size() && i < kMaxRows; i++) {
        const char* name = "?";
        if (ActorDB::Instance != nullptr) {
            ActorDB::Entry& e = ActorDB::Instance->RetrieveEntry(rows[i].first);
            if (e.entry.valid && !e.name.empty()) {
                name = e.name.c_str();
            }
        }
        f32 avg = (f32)rows[i].second / (f32)sCensusFrames;
        char line[128];
        snprintf(line, sizeof(line), "%5.1f  %s (id %d)\n", avg, name, (int)rows[i].first);
        sCensusText += line;
    }
    if (rows.size() > kMaxRows) {
        char more[64];
        snprintf(more, sizeof(more), "... and %d more ids\n", (int)(rows.size() - kMaxRows));
        sCensusText += more;
    }

    sCensusAccum.clear();
    sCensusFrames = 0;
}

static void RefreshFrameParams() {
    sParams.cel = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ToonLighting.Enabled"), 1) != 0;
    // One selector drives all three systems, so they can never both draw. Values outside the enum (a
    // hand-edited config, or a future mode read by an older build) fall back to vanilla rather than
    // leaving every system off.
    const int mode = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode"), SHADOW_MODE_VANILLA);
    sParams.shadowMode = (mode >= SHADOW_MODE_VANILLA && mode <= SHADOW_MODE_SHADOW_MAP) ? mode : SHADOW_MODE_VANILLA;
    sParams.shadows = sParams.shadowMode == SHADOW_MODE_ACTOR;
    sParams.shadowMap = sParams.shadowMode == SHADOW_MODE_SHADOW_MAP;
    // Only worth asking when the mode is actually selected; otherwise leave it false so the per-actor
    // getters short-circuit on the cheap check.
    // SOH [Enhancement] GPU profiling of the shadow pass, which is a plain flag on the backend and changes
    // nothing about what is drawn. Pushed from out here rather than from inside the branch below so it is
    // also CLEARED when the shadow map is switched off -- otherwise the timer would keep opening queries
    // every frame for a pass that no longer runs, and report on nothing.
    if (Fast::GfxRenderingAPI* profileRapi = GetRenderingApi(); profileRapi != nullptr) {
        profileRapi->SetShadowMapProfiling(sParams.shadowMap &&
                                           CVarGetInteger(CVAR_DEVELOPER_TOOLS("ShadowMap.ProfileGpu"), 0) != 0);
        // The depth map itself, drawn in a corner. Every other shadow diagnostic here asks what the SHADING
        // pixel was handed; this one shows what the depth pass STORED, which until now nothing could.
        profileRapi->SetShadowMapViewSlice(
            sParams.shadowMap ? CVarGetInteger(CVAR_DEVELOPER_TOOLS("ShadowMap.ViewSlice"), 0) : 0);
    }
    if (sParams.shadowMap) {
        Fast::GfxRenderingAPI* rapi = GetRenderingApi();
        sParams.shadowMapSupported = rapi != nullptr && rapi->SupportsShadowMap();
        // Caster reach = the last active cascade's far split (see shadowMapReach).
        static const char* kSplitCVars[SHADOW_MAP_MAX_CASCADES] = { CVAR_ENHANCEMENT("Graphics.ShadowMap.Split0"),
                                                                    CVAR_ENHANCEMENT("Graphics.ShadowMap.Split1"),
                                                                    CVAR_ENHANCEMENT("Graphics.ShadowMap.Split2") };
        static const f32 kSplitDefaults[SHADOW_MAP_MAX_CASCADES] = { SHADOW_MAP_DEFAULT_SPLIT_0,
                                                                     SHADOW_MAP_DEFAULT_SPLIT_1,
                                                                     SHADOW_MAP_DEFAULT_SPLIT_2 };
        s32 count = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.CascadeCount"), SHADOW_MAP_DEFAULT_CASCADES);
        count = count < 1 ? 1 : (count > SHADOW_MAP_MAX_CASCADES ? SHADOW_MAP_MAX_CASCADES : count);
        sParams.shadowMapReach = CVarGetFloat(kSplitCVars[count - 1], kSplitDefaults[count - 1]);
        sParams.shadowMapCasterFirst = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.CasterFirst"), 0) != 0;
        sParams.shadowMapCasterDrawRadius = CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.CasterDrawRadius"),
                                                         SHADOW_MAP_DEFAULT_CASTER_DRAW_RADIUS);
    } else {
        sParams.shadowMapSupported = false;
        sParams.shadowMapCasterDrawRadius = 0.0f;
    }
    sParams.suppressVanilla = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.SuppressVanillaShadows"), 1) != 0;
    sParams.multipleLights = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ToonLighting.MultipleLights"), 1) != 0;
    sParams.localIntensity = std::clamp(
        CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ToonLighting.LocalIntensity"), 0.5f), 0.0f, 1.0f);
    sParams.useNaviLight = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ToonLighting.UseNaviLight"), 1) != 0;
    sParams.showDebug = CVarGetInteger(CVAR_DEVELOPER_TOOLS("ToonLighting.ShowDebug"), 0) != 0;
    sParams.shadowMapCensus =
        sParams.shadowMap && CVarGetInteger(CVAR_DEVELOPER_TOOLS("ShadowMap.ShowCascadeBounds"), 0) != 0;
    ToonShadowCensusPublish();
    sParams.pointRange =
        CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ToonLighting.PointLightRange"), kDefaultPointLightRange);
    sParams.transitionTime =
        CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ToonLighting.TransitionTime"), kDefaultTransitionTime);
    sParams.maxDist =
        (f32)CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.MaxDistance"), kDefaultShadowMaxDistance);
}

// Frame-constant easing terms (they depend only on R_UPDATE_RATE and the transition-time CVar, so
// computing them per actor repaid an expf for every drawn actor every frame). Refreshed in
// OnToonFrameUpdate alongside sParams.
static f32 sToonKeyDt = 3.0f / 60.0f; // seconds per game draw
static f32 sToonKeyAlpha = 0.2f;      // per-draw slerp fraction; reaches ~99% in transitionTime seconds

// Navi's two emitted lights, resolved once per frame when the player opted her out of key selection
// (identical for every actor; see ToonClosestPointLight). Compared by address only, never dereferenced.
static LightInfo* sNaviGlow = NULL;
static LightInfo* sNaviNoGlow = NULL;

// C-callable getters (see ToonLighting.h): the decompiled draw code asks these instead of doing its own
// per-actor CVar lookups, and they keep the bracket/hook decisions consistent within a frame.
extern "C" int ToonLighting_FeaturesActive(void) {
    return sParams.cel || sParams.shadows || sParams.shadowMap;
}
extern "C" int ToonLighting_CelEnabled(void) {
    return sParams.cel;
}
extern "C" int ToonLighting_ShadowsEnabled(void) {
    return sParams.shadows;
}
// Whether the running backend can actually draw shadow maps -- answered from the once-per-frame snapshot,
// never queried live: reaching the backend means a dynamic_pointer_cast plus a weak_ptr lock, and this is
// read from the per-actor path. The cascaded depth pass is Direct3D 11 only; OpenGL and Metal have no
// implementation, and there the mode must degrade to vanilla rather than suppressing the vanilla shadows
// and drawing nothing in their place. It is a runtime query rather than an #ifdef because one binary can
// pick its backend at launch.
extern "C" float ToonLighting_ShadowMapCasterDrawRadius(void) {
    return (sParams.shadowMap && sParams.shadowMapSupported) ? sParams.shadowMapCasterDrawRadius : 0.0f;
}
// SOH [Enhancement] Does the frame reorder apply this frame? See sParams.shadowMapCasterFirst and the
// block it gates in z_play.c. False whenever shadow-map mode is not actually running, so the reorder can
// never happen without the thing it exists for.
extern "C" int ToonLighting_ShadowMapCasterFirst(void) {
    return (sParams.shadowMap && sParams.shadowMapSupported && sParams.shadowMapCasterFirst) ? 1 : 0;
}

extern "C" int ToonLighting_ShadowMapEnabled(void) {
    return sParams.shadowMap && sParams.shadowMapSupported;
}

// Is this caster worth drawing purely so it can cast? Shared by the actor culling and the room-mesh shape
// culling, because it is one question asked in two places.
//
// The game's own culling is a screen test and drops everything behind the camera, so both call sites need an
// escape -- and the escape used to be a sphere around the camera. A sphere is the wrong shape. What decides
// whether a caster matters is not how far away it stands but whether its SHADOW arrives, and a low sun sends
// a shadow much further than the caster stands: the Kakariko windmill can sit well outside any sphere worth
// paying for and still lay its shadow across the ground at the player's feet. Growing the sphere until it
// reached would mean drawing everything in every direction to catch the few things lying along one of them.
//
// So follow the light. The shadow of this caster runs from it along the light ray, for as far as a caster of
// SHADOW_MAP_CASTER_SHADOW_HEIGHT would reach at the current elevation -- which is the whole point of doing
// it this way, since that length grows exactly when shadows stretch and collapses at midday when they do
// not. Keep the caster when any point of that ray passes within reach of the camera. The far tower whose
// shadow arrives is admitted; the equally far one whose shadow goes the other way still is not.
//
// casterSize is how far the caster's geometry reaches past the point given, so a large object is measured by
// its surface rather than by its origin.
extern "C" int ToonLighting_ShadowCasterInReach(f32 x, f32 y, f32 z, f32 casterSize) {
    if (!sParams.shadowMap || !sParams.shadowMapSupported || gPlayState == NULL) {
        return 0;
    }
    f32 radius = sParams.shadowMapCasterDrawRadius;
    if (radius <= 0.0f) {
        return 0;
    }
    f32 reach = radius + (casterSize > 0.0f ? casterSize : 0.0f);

    // Camera relative to the caster, world space. Both ends are world positions on purpose: the projected
    // coordinates the surrounding culling uses carry the projection's scale factors in x and y, and only
    // their z is a distance.
    f32 vx = gPlayState->view.eye.x - x;
    f32 vy = gPlayState->view.eye.y - y;
    f32 vz = gPlayState->view.eye.z - z;

    const f32* dir = sParams.shadowMapLightDir;
    // How far along the ray a shadow can run. The light's y component IS the sine of its elevation (the
    // vector is normalised), and a caster of height H drops that much over H/sin -- so this is the
    // parametric length of the tallest shadow the scene can throw right now. Guarded because a light on the
    // horizon would divide by nothing; the elevation floor keeps it near 0.5 in practice.
    f32 elev = dir[1] < 0.0f ? -dir[1] : dir[1];
    f32 rayLen = (elev > 0.01f) ? (SHADOW_MAP_CASTER_SHADOW_HEIGHT / elev) : 0.0f;

    f32 t = (vx * dir[0]) + (vy * dir[1]) + (vz * dir[2]);
    if (t < 0.0f) {
        t = 0.0f; // behind the caster along the light: the nearest point of the shadow is the caster itself
    } else if (t > rayLen) {
        t = rayLen;
    }
    f32 ox = vx - (dir[0] * t);
    f32 oy = vy - (dir[1] * t);
    f32 oz = vz - (dir[2] * t);
    return (((ox * ox) + (oy * oy) + (oz * oz)) < (reach * reach)) ? 1 : 0;
}
extern "C" int ToonLighting_ShadowMode(void) {
    return sParams.shadowMode;
}
extern "C" int ToonLighting_SuppressVanillaShadows(void) {
    // Only a mode that actually draws may hide the vanilla shadows -- otherwise picking Shadow Map on a
    // backend without one would leave the scene with no shadows at all.
    return (sParams.shadows || ToonLighting_ShadowMapEnabled()) && sParams.suppressVanilla;
}

// Actors the cel system skips entirely: they look wrong relit AND wrong casting a flattened shadow
// (doors, the Great Deku Tree, water-box surfaces). Data-driven so it's easy to extend after seeing what
// a scene actually relights -- add an ACTOR_* id here (or a category below). The wooden sign (En_Kanban)
// is intentionally NOT excluded: its shape shadow is exactly what this feature is meant to reproduce.
static bool ToonActorExcluded(Actor* actor) {
    if (actor->category == ACTORCAT_DOOR) {
        return true; // every door variant in one check
    }
    switch (actor->id) {
        case ACTOR_BG_TREEMOUTH:  // Great Deku Tree (very tall)
        case ACTOR_BG_MIZU_WATER: // water-box surfaces
        case ACTOR_BG_HAKA_WATER:
        case ACTOR_EN_WOOD02:  // trees / bushes / leaf scenery
        case ACTOR_OBJ_SWITCH: // floor/crystal/eye switches -- environment fixtures, not relit objects.
        case ACTOR_OBJ_BEAN:   // magic bean plant/platform -- same. Both are also RECEIVERS below, so
                               // they still catch other actors' shadows like the ground does.
            return true;
        default:
            break;
    }
    // All Bg_Spot* overworld scenery (bridges, fences, gates, rocks, well/oasis water, ...) reads as part of the
    // environment, not a relit actor. Matched by name prefix so every Bg_Spot variant is covered without listing
    // ~30 scattered actor IDs. RetrieveEntry is bounds-safe and returns an empty name for unknown ids.
    // The verdict is cached per id (this runs for every drawn actor every frame; ids are stable, so the
    // string lookup+compare only ever happens once per actor type).
    if (ActorDB::Instance != nullptr) {
        static std::unordered_map<s32, bool> sBgSpotVerdicts;
        auto [it, isNew] = sBgSpotVerdicts.try_emplace((s32)actor->id, false);
        if (isNew) {
            it->second = ActorDB::Instance->RetrieveEntry(actor->id).name.rfind("Bg_Spot", 0) == 0;
        }
        return it->second;
    }
    return false;
}

// Actors whose model geometry extends well BELOW the floor (a signpost's post is buried in the ground). The
// shadow slab is built at the lowest captured vertex, so for these the slab would sit underground and never
// reach the visible floor. Flagging them tells the renderer to lift the slab's feet up to the floor Y (passed
// as the clamp), putting the shadow back on the ground. Extend with other deep-rooted props as they turn up.
static bool ToonShadowDeepRooted(Actor* actor) {
    return actor->id == ACTOR_EN_KANBAN; // wooden signposts
}

// Actors that are really SCENERY. A tree is spawned as an actor, but nothing about it is a character: it
// stands still, it is part of the place, and its shadow belongs on everything the way a wall's does. That
// distinction decides which caster layer it goes in, and the layers are not interchangeable -- characters
// deliberately skip the actor layer so they cannot shadow each other, so a tree armed as an actor casts onto
// the ground and never onto the player standing under it. Named here, it goes in the world layer instead,
// beside the room mesh, and everything samples that.
//
// Bracketing them also reaches geometry the actor arming never could. A tree is two draws: EnWood02_Draw
// sends the trunk to POLY_OPA and the leaves to POLY_XLU, because the leaves fade with distance through an
// env-colour alpha. Every actor marker goes into POLY_OPA and the two buffers run opaque-first, so the
// canopy was never inside a bracket at all -- a tree cast its trunk and nothing else.
//
// A whitelist rather than a render-mode test, because no render-mode test can answer either question. The
// canopy sets exactly the state a pane of glass sets -- translucent zmode, alpha compare usually off -- so a
// rule wide enough to admit it would also admit every water plane, glow, aura and particle in the
// translucent stream, each of which would then drop a solid shadow. What makes a tree different is knowledge
// about the model, so the knowledge is written down here. Its translucent half still only ever casts through
// the alpha cutout: with no usable texture to cut against, a bracketed draw casts nothing.
static bool ToonShadowSceneryCaster(Actor* actor) {
    // Asked the other way round: not "is this scenery" but "is this a CHARACTER".
    //
    // Listing the scenery categories was the wrong shape and it kept missing things -- name BG, DOOR and
    // SWITCH and a gate filed under PROP still falls through, and there is no way to know which ones are
    // missing except by finding each of them in a screenshot. The character list has no such problem,
    // because the actor caster layer exists for exactly one reason: characters must not shadow each other.
    // What belongs in it is precisely the things that rule is about, and that is these four and nothing else.
    //
    // Everything else the game spawns is part of the place -- gates, doors, chests, pots, bombs, signs,
    // trees, switches -- and its shadow belongs on everything, the way a wall's does. Anything in that set
    // that genuinely should not cast is named in ToonShadowExcluded, which is where Navi and the water
    // surfaces already live; that list is the exception and this is the rule, rather than the other way
    // round.
    switch (actor->category) {
        case ACTORCAT_PLAYER:
        case ACTORCAT_NPC:
        case ACTORCAT_ENEMY:
        case ACTORCAT_BOSS:
            return false;
        default:
            return true;
    }
}

// Tracks whether the scenery bracket is currently open in the stream, so the markers are emitted only on a
// change. Most frames contain no scenery actor at all and then this costs nothing.
static bool sSceneryCasterArmed = false;

// The caster marker is NOT mirrored into the translucent display list, and that is deliberate rather than an
// omission. It was, briefly: the Hyrule Castle gate is one of over a hundred actors drawn through
// Gfx_SetupDL_25Xlu, which appends to POLY_XLU while setting an OPAQUE zmode, so a marker that only reached
// POLY_OPA could never see them. Scenery reaches its own bracket now (ToonShadowSceneryCaster, which covers
// both lists), so the only actors left taking the marker below are characters -- and what a character keeps
// in the translucent list is never its body. It is effects: a charge trail, a hover-boots ring, a frozen
// shell, a get-item sparkle. Those declare an opaque zmode as readily as the gate does, so the renderer
// cannot turn them away on render state, and Link's spin-attack trail was landing on the ground as a set of
// solid black blades.
//
// So characters cast from their opaque geometry only. A character with real solid geometry queued in the
// translucent list would lose that part of its shadow; that is a far smaller loss than a glow casting one.

// Walkable "floor" actors: scenery the player stands on that the game spawns as actors rather than baking
// into the room mesh (the castle-town drawbridge, the Gerudo Valley bridge, some dungeon platforms). Because
// they are actors they are normally drawn AFTER the shadow-volume flush and so receive no shadow. The actor
// draw loop (func_800315AC) special-cases this set: it draws them BEFORE the flush so their surfaces are in
// the depth buffer and catch shadows like the static scene. Curated by id on purpose -- only flat, broadly
// static, genuinely-walked-on pieces belong here (a moving platform would show the shadow's one-frame lag).
// Extend cautiously and verify per actor; candidates to try next are noted inline.
// NOTE: the receiver pre-pass in z_actor.c only scans the BG, PROP and SWITCH actor lists -- every id
// below is in one of those categories. If a receiver from another category is ever added here, extend
// that scan or the new receiver will silently never pre-draw.
static bool ToonShadowReceiver(Actor* actor) {
    switch (actor->id) {
        case ACTOR_BG_SPOT00_HANEBASI: // Hyrule Field <-> Castle Town drawbridge (the planks you cross)
        case ACTOR_BG_SPOT09_OBJ:      // Gerudo Valley rope bridge
        case ACTOR_BG_MORI_BIGST:      // Forest Temple falling platform (Stalfos room); static until it drops
        case ACTOR_BG_HAKA_MEGANEBG:   // Shadow Temple lens-revealed stone platforms (static)
        case ACTOR_BG_MENKURI_KAITEN:  // Large rotating stone ring (Gerudo Training Ground + Forest Temple).
                                       // Genuinely rotates while ridden, so the shadow shows a one-frame lag
                                       // during motion -- the test case for whether moving receivers look OK.
        case ACTOR_OBJ_SWITCH:         // floor switches are stood on (SWITCH category -- see the pre-pass note)
        case ACTOR_OBJ_BEAN:           // the bean platform is ridden; excluded from relight too (above)
            return true;
        case ACTOR_BG_HAKA_GATE: {
            // Shadow Temple. One overlay drives four different things; the variant is the low byte of params
            // (the high byte is a switch flag the actor's Init strips). The overlay's own enum is
            // STATUE=0, FLOOR=1, GATE=2, SKULL=3. The walkable surfaces are the round opening trap FLOOR (1) and
            // the truth-spinner STATUE disc (0) you stand on -- both should catch shadows + light; verified in
            // the trap-floor room. GATE (2) and SKULL (3) stay non-receivers so the skull posts around it keep
            // casting their own shadows.
            u16 type = actor->params & 0xFF;
            return type == 0 || type == 1;
        }
        // Further candidates (enable + test individually; some move, so watch the one-frame lag):
        //   ACTOR_BG_HIDAN_SIMA (Fire Temple stone platform), ACTOR_BG_DDAN_JD (Dodongo rising platform),
        //   ACTOR_BG_JYA_KANAAMI (Spirit grating bridge).
        default:
            return false;
    }
}

// Actors that keep cel relight but should NOT cast a drop shadow (unlike ToonActorExcluded, which drops both) --
// rationale per id inline. Shadow receivers are excluded too: a walkable floor casting its own silhouette down
// into the void below reads wrong, and (now that it sits in the depth buffer at flush time) could self-shadow.
static bool ToonShadowExcluded(Actor* actor) {
    switch (actor->id) {
        // Actors that are pure light, named one at a time. The shortcut is a trap and it is worth recording
        // why, because it looks compelling: carve out the CATEGORIES the game files effects under. It does
        // not work here. ACTORCAT_ITEMACTION holds En_M_Thunder and the ocarina effects, and it also holds
        // Bg_Gate_Shutter and Door_Gerudo -- solid gates whose shadows were the point of several fixes
        // before this one. ACTORCAT_MISC holds magic and flames, and also Bg_Sst_Floor, a boss arena's
        // floor. The categories sort actors by what the engine does with them, not by what they are made
        // of, and the distinction this needs is the second one.
        //
        // They have to be named at all because the rule above is that scenery is the DEFAULT, with only
        // characters carved out of it -- so an effect actor lands in the world caster layer beside the room
        // mesh unless something stops it. That is what put the charged spin ring's own silhouette across
        // the ground it was lighting.
        case ACTOR_EN_M_THUNDER: // spin attack charge and its release
        case ACTOR_MAGIC_DARK:   // Nayru's Love
        case ACTOR_MAGIC_FIRE:   // Din's Fire
        case ACTOR_MAGIC_WIND:   // Farore's Wind
        case ACTOR_EN_ICE_HONO:  // blue fire
        case ACTOR_EN_M_FIRE1:   // flames
        // Navi and every other fairy. She is a light SOURCE -- the world-lighting feature casts a pool from
        // her -- so a shadow off her is wrong on its own terms before any render state is consulted.
        //
        // She also slips the renderer's translucency test, which is worth writing down because the next glow
        // will do the same. That test turns away ZMODE_XLU, and Navi's body is drawn with the cloud-surface
        // blender: alpha-blended through and through, and declared ZMODE_OPA. The zmode field is what the
        // engine says about depth sorting, not about opacity, and the two part company exactly here. She was
        // harmless while the caster marker only reached the opaque display list; she is in the translucent
        // one, and now that the marker reaches there too she has to be named.
        case ACTOR_EN_ELF:
        case ACTOR_EN_KUSA: // small cuttable grass -- everywhere and tiny, a blob per tuft reads wrong
        // Ambient critters. Same argument as the grass: sprite-sized, spawned in swarms (Obj_Mure drops
        // them a dozen at a time across a field), and every one of them lands in the world caster layer by
        // default, so a meadow comes out speckled with little dark diamonds that track nothing the player
        // can see. They are ambience, not scenery.
        case ACTOR_EN_BUTTE:  // butterflies
        case ACTOR_EN_INSECT: // bugs
        case ACTOR_EN_FISH:   // fish
        // The environment particle swarm: fairies, snow, lightning, the Sun's Song grave sparks, and the
        // shafts of light through the trees. One actor, drawing dozens of billboards, which is why it is
        // the thing the census could name and reasoning could not -- it reads as a SINGLE scenery caster
        // and paints a whole field with bars.
        //
        // Every one of its five modes is light or weather. Not one of them is an object.
        case ACTOR_OBJECT_KANKYO:
        case ACTOR_EN_SKJ:       // Skull Kid
        case ACTOR_EN_DNT_NOMAL: // Deku Scrub mound dwellers
        case ACTOR_EN_KZ:        // King Zora
        // Water surfaces. These sit in ToonActorExcluded for their own reasons, and that list no longer
        // implies "does not cast" -- so they have to be named here or a water box, which is a single flat
        // plane spanning a whole room, would drop the entire volume beneath it into shadow.
        case ACTOR_BG_MIZU_WATER:
        case ACTOR_BG_HAKA_WATER:
            return true;
        default:
            return ToonShadowReceiver(actor);
    }
}

// The scenery-caster census as a printable table, for the debug section of the menu. Empty string when the
// debug view is off (nothing is being counted then) or before the first window has closed.
extern "C" const char* ToonLighting_ShadowMapCasterCensus(void) {
    return sCensusText.c_str();
}

// C-callable export (see ToonLighting.h): lets the decompiled actor draw loop reorder receivers ahead of the
// shadow flush without pulling the curated id list into the game code.
// SOH [Enhancement] What the cascades actually came out as, for the menu to print.
//
// Read from the renderer rather than recomputed from the CVars, and that is the whole point: with the
// automatic ladder on, the split sliders no longer describe where the bands are, and the texel size never
// did -- it falls out of the projection the fit builds, which depends on the camera. Recomputing either
// here would produce a table that can disagree with the picture.
//
// Rebuilt into a static string on each call. The menu asks once per frame while its page is open, which is
// nothing next to what a frame already does.
extern "C" const char* ToonLighting_ShadowMapCascadeReport(void) {
    static std::string report;
    report.clear();

    Fast::GfxRenderingAPI* rapi = GetRenderingApi();
    if (rapi == nullptr) {
        report = "Renderizador indisponível.";
        return report.c_str();
    }
    float splits[SHADOW_MAP_MAX_CASCADES] = {};
    float texels[SHADOW_MAP_MAX_CASCADES] = {};
    const int count = rapi->ShadowMapCascadeReport(splits, texels, SHADOW_MAP_MAX_CASCADES);
    if (count <= 0) {
        report = "Nenhuma cascata neste quadro.";
        return report.c_str();
    }

    const f32 blend =
        CLAMP(CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.BlendFraction"), SHADOW_MAP_DEFAULT_BLEND_FRACTION),
              0.0f, 1.0f);
    char line[192];
    for (int i = 0; i < count; i++) {
        const float nearEdge = (i == 0) ? 0.0f : splits[i - 1];
        const float farEdge = splits[i];
        // The band the shader cross-fades over, in the same terms it computes it: a fraction of THIS
        // cascade's own span, measured back from its far edge.
        const float fadeStart = farEdge - ((farEdge - nearEdge) * blend);
        snprintf(line, sizeof(line), "Faixa %d:  %.0f a %.0f  |  texel %.2f un.  |  transição %.0f a %.0f\n", i + 1,
                 nearEdge, farEdge, texels[i], fadeStart, farEdge);
        report += line;
    }
    snprintf(line, sizeof(line), "Alcance total: %.0f unidades", splits[count - 1]);
    report += line;
    return report.c_str();
}

extern "C" int ToonLighting_IsShadowReceiver(Actor* actor) {
    return (actor != nullptr && ToonShadowReceiver(actor)) ? 1 : 0;
}

// Tracks whether the toon (cel-relight) bracket is currently ON in the display-list stream. The actor-loop
// bracket (func_800315AC) opens it ON; HandleActorDraw flips it OFF around blacklisted actors and back ON
// for the next normal actor, deduped so same-state runs emit nothing. Reset each frame to match the
// bracket. Only meaningful while cel shading is enabled (the only thing that opens the bracket).
static bool sToonEnabled = true;

// The Fast3D rendering backend, if the active window is the Fast3D window. Null on other windows
// (e.g. headless), in which case there is nothing to relight and pushing is simply skipped.
static Fast::GfxRenderingAPI* GetRenderingApi() {
    auto wnd = std::dynamic_pointer_cast<Fast::Fast3dWindow>(Ship::Context::GetInstance()->GetWindow());
    if (wnd == nullptr) {
        return nullptr;
    }
    auto interpreter = wnd->GetInterpreterWeak().lock();
    if (interpreter == nullptr) {
        return nullptr;
    }
    return interpreter->GetCurrentRenderingAPI();
}

// The Fast3D interpreter itself (for actor-shadow config, which lives in the interpreter rather than the
// backend). Null on non-Fast3D/headless windows, where there is nothing to draw.
static std::shared_ptr<Fast::Interpreter> GetInterpreter() {
    auto wnd = std::dynamic_pointer_cast<Fast::Fast3dWindow>(Ship::Context::GetInstance()->GetWindow());
    if (wnd == nullptr) {
        return nullptr;
    }
    return wnd->GetInterpreterWeak().lock();
}

// The last toon key emitted this pass, as the quantized bytes gSPToonKey carries (s8 dir, u8 color).
// An actor whose key quantizes to the same bytes skips re-emitting, so same-key actors (e.g. everything
// lit by the sun) need no per-object flush and batch together. Cleared each frame in OnToonFrameUpdate.
static bool sHaveLastKey = false;
static s8 sLastKeyDir[3];
static u8 sLastKeyCol[3];

static void ToonClearKeyStates(); // defined with the key-state map below

// Runs once per frame (game-frame-update hook, after the frame's draw). Pushes the frame-global ramp
// shape to the renderer and clears the per-pass key-dedup state so the next frame's first actor
// re-emits. The ramp is pure look-tuning, so it lives in SoH's CVars; reading it here (once per frame)
// keeps those strings out of the framework.
static void OnToonFrameUpdate() {
    // Refresh the per-frame CVar snapshot first: everything below (and every HandleActorDraw this
    // frame) reads the cached values, so a toggle from the menu OR the console lands within a frame.
    RefreshFrameParams();
    // Clear before any early-out, so the dedup state resets even on a headless window (no renderer).
    sHaveLastKey = false;
    // The bracket re-opens toon ON each frame; match it so the first blacklisted actor toggles correctly.
    sToonEnabled = true;
    if (!sParams.cel && !sParams.shadows && !sParams.shadowMap) {
        // Every feature off (possibly via console, which never re-runs RegisterToonLighting): nothing
        // below is needed, and the eased per-actor state must not go stale while disabled.
        // shadowMap has to be in this test -- the cascade configuration is pushed further down, and
        // returning early here left it stale whenever Shadow Map was on with cel shading off.
        ToonClearKeyStates();
        return;
    }

    // Frame-constant easing terms (see the statics above).
    sToonKeyDt = (R_UPDATE_RATE > 0 ? R_UPDATE_RATE : 3) / 60.0f;
    {
        f32 tt = sParams.transitionTime < 0.05f ? 0.05f : sParams.transitionTime;
        sToonKeyAlpha = 1.0f - expf(-4.6f * sToonKeyDt / tt);
    }

    // Navi's opt-out lights (see the statics above). Identification matches the light-casting feature:
    // player->naviActor, an En_Elf with FAIRY_NAVI params.
    sNaviGlow = sNaviNoGlow = NULL;
    if (!sParams.useNaviLight && gPlayState != NULL) {
        Player* player = GET_PLAYER(gPlayState);
        if ((player != NULL) && (player->naviActor != NULL) && (player->naviActor->id == ACTOR_EN_ELF) &&
            (player->naviActor->params == FAIRY_NAVI)) {
            EnElf* navi = (EnElf*)player->naviActor;
            sNaviGlow = &navi->lightInfoGlow;
            sNaviNoGlow = &navi->lightInfoNoGlow;
        }
    }

    // Actor shadow look tuning (global, not per object): core blend strength, the slab depth/rise that bound
    // the conforming ground band, and a "length" slider mapped to the minimum grazing angle that bounds how
    // far a low-angle key may stretch the cast shadow. Lives in the interpreter (it owns the shadow
    // projection), pushed here.
    if (auto interp = GetInterpreter()) {
        f32 opacity = CVarGetFloat(CVAR_ENHANCEMENT("Graphics.WorldShadows.Opacity"), kDefaultShadowOpacity);
        f32 length = CVarGetFloat(CVAR_ENHANCEMENT("Graphics.WorldShadows.Length"), kDefaultShadowLength);
        f32 slabDepth = CVarGetFloat(CVAR_ENHANCEMENT("Graphics.WorldShadows.SlabDepth"), kDefaultShadowSlabDepth);
        f32 slabRise = CVarGetFloat(CVAR_ENHANCEMENT("Graphics.WorldShadows.SlabRise"), kDefaultShadowSlabRise);
        s32 edgeSoftness =
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.EdgeSoftness"), kDefaultShadowEdgeSoftness);
        bool showVolume = CVarGetInteger(CVAR_DEVELOPER_TOOLS("WorldShadows.ShowVolume"), 0);
        // Map the Length slider to how steeply the key light is forced before projecting: low Length = steep
        // (short shadow tucked under the actor), high Length = lets a low key cast a long lean. 0 => 0.95, 1 => 0.10.
        f32 minElevation = 0.95f - (CLAMP(length, 0.0f, 1.0f) * 0.85f);
        // Slab Depth/Rise: how far below/above the feet the stencil volume reaches (the band of ground it conforms to).
        interp->SetToonShadowParams(opacity, minElevation, slabDepth, slabRise, edgeSoftness, showVolume);

        // Cascaded shadow maps. `enabled` already folds in the backend capability, so the interpreter can
        // trust it and does not repeat the check. The key light direction is the same one the cel shading
        // picks, negated: the toon uniform points from the surface toward the light, while a shadow
        // projection needs the direction the light travels.
        const bool shadowMapOn = ToonLighting_ShadowMapEnabled() != 0;
        f32 splits[SHADOW_MAP_MAX_CASCADES] = {
            CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.Split0"), SHADOW_MAP_DEFAULT_SPLIT_0),
            CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.Split1"), SHADOW_MAP_DEFAULT_SPLIT_1),
            CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.Split2"), SHADOW_MAP_DEFAULT_SPLIT_2)
        };
        // Per-cascade update rate, as a divisor of the frame rate: 1 rebuilds every frame, 2 every other.
        // At 60 fps the defaults run the near and mid cascades at 60 Hz and the far one at 30 Hz -- the far
        // one is the cheapest to halve, since it covers the most ground per texel and so changes the least
        // between frames, while sweeping in the most casters to draw.
        int cascadeDivisors[SHADOW_MAP_MAX_CASCADES] = {
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor0"), SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_0),
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor1"), SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_1),
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor2"), SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_2)
        };
        for (int i = 0; i < SHADOW_MAP_MAX_CASCADES; i++) {
            cascadeDivisors[i] = CLAMP(cascadeDivisors[i], 1, SHADOW_MAP_MAX_CASCADE_DIVISOR);
        }
        // One sun (or moon) for the whole frame, straight from the environment directionals. This used to
        // take the last actor key emitted, which looked right until Navi walked on screen: Navi IS a light,
        // so her key became "the frame's light" and every shadow in the scene swung to point away from her.
        // A single set of cascades can only have one direction, and it has to be the one that is actually
        // global.
        f32 envDir[3] = { 0.0f, 1.0f, 0.0f };
        f32 envCol[3] = { 1.0f, 1.0f, 1.0f };
        if (gPlayState != NULL) {
            ToonEnvKey(gPlayState, envDir, envCol);
        }
        // Lift the light off the horizon before using it. A near-horizontal sun stretches every shadow
        // towards infinity, which looks wrong well before it is geometrically wrong, and it also wastes the
        // cascades on a footprint far longer than the scene being shaded. The compass bearing is kept; only
        // the height is raised, and only when it is actually too low: a floor, not a remap. The stencil
        // volumes remap instead, which is continuous but also shortens midday shadows -- at 45 degrees it
        // would cut them by nearly half. A depth map can be faithful above the threshold and only intervene
        // below it, so that is what this does.
        // Same shape as the stencil volumes' Length control, but with its own value: a depth map can afford
        // longer shadows than a flattened silhouette can.
        f32 minElev =
            CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.MinElevation"), SHADOW_MAP_DEFAULT_MIN_ELEVATION);
        minElev = CLAMP(minElev, 0.05f, 0.99f);
        // Negated on the way out: the toon convention points from the surface toward the light, while a
        // shadow projection needs the direction the light travels.
        f32 lightDir[3] = { 0.0f, -1.0f, 0.0f };
        {
            const f32 up = envDir[1] < 0.0f ? 0.0f : envDir[1]; // below the horizon reads as "on it"
            const f32 elev = up < minElev ? minElev : up;
            const f32 hLen = sqrtf((envDir[0] * envDir[0]) + (envDir[2] * envDir[2]));
            if (hLen >= 0.001f) {
                const f32 hScale = sqrtf(1.0f - (elev * elev)) / hLen;
                lightDir[0] = -hScale * envDir[0];
                lightDir[1] = -elev;
                lightDir[2] = -hScale * envDir[2];
            }
            // hLen ~ 0 means the light is straight overhead: the default straight-down vector is right.
        }
        // Keep it for the caster-reach test, which follows a shadow along this ray (see
        // ToonLighting_ShadowCasterInReach). Stored after the elevation floor, so it is the same light the
        // cascades are actually built from and not the raw environment one.
        sParams.shadowMapLightDir[0] = lightDir[0];
        sParams.shadowMapLightDir[1] = lightDir[1];
        sParams.shadowMapLightDir[2] = lightDir[2];
        // SOH [Enhancement] Edge quality (see fast/shadow_map.h and the "Qualidade das Sombras" menu tab).
        // Five independent techniques that shape the shadow's EDGE; everything else in this block decides
        // where the shadow falls. Pushed before the params below so the split ladder -- which is one of the
        // five and is fitted on the CPU -- is in effect for the splits that arrive with them.
        {
            ShadowMapQuality quality = ShadowMapQualityDefaults();
            quality.analyticEdge = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdge"), 0);
            quality.analyticEdgeWidth = CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdgeWidth"),
                                                     SHADOW_MAP_DEFAULT_ANALYTIC_EDGE_WIDTH);
            quality.jitter = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Jitter"), 0);
            quality.jitterTaps =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterTaps"), SHADOW_MAP_DEFAULT_JITTER_TAPS);
            quality.jitterRadius =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterRadius"), SHADOW_MAP_DEFAULT_JITTER_RADIUS);
            quality.layout =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Layout"), SHADOW_MAP_DEFAULT_LAYOUT);
            quality.staticCache =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.StaticCache"), SHADOW_MAP_DEFAULT_STATIC_CACHE);
            quality.clipmapLevels = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapLevels"),
                                                   SHADOW_MAP_DEFAULT_CLIPMAP_LEVELS);
            quality.clipmapBase =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapBase"), SHADOW_MAP_DEFAULT_CLIPMAP_BASE);
            quality.clipmapResolution = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapResolution"),
                                                       SHADOW_MAP_DEFAULT_CLIPMAP_RESOLUTION);
            quality.edgeHarden =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHarden"), SHADOW_MAP_DEFAULT_EDGE_HARDEN);
            quality.edgeHardness =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHardness"), SHADOW_MAP_DEFAULT_EDGE_HARDNESS);
            quality.edgeThreshold = CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeThreshold"),
                                                 SHADOW_MAP_DEFAULT_EDGE_THRESHOLD);
            quality.ladderMode =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderMode"), SHADOW_MAP_DEFAULT_LADDER_MODE);
            quality.ladderLambda =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderLambda"), SHADOW_MAP_DEFAULT_LADDER_LAMBDA);
            quality.ladderNear =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderNear"), SHADOW_MAP_DEFAULT_LADDER_NEAR);
            // SOH [Enhancement] Shadow acne (see fast/shadow_map.h and the "Correção de Acne" menu tab).
            // Defaults come from the header rather than being written twice, so the menu's "restore
            // defaults" and the renderer's fallback cannot drift apart.
            const ShadowMapAcne acneDefaults = ShadowMapQualityDefaults().acne;
            quality.acne.enabled =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.Enabled"), acneDefaults.enabled);
            quality.acne.normalOffset =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalOffset"), acneDefaults.normalOffset);
            quality.acne.normalTexels =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalTexels"), acneDefaults.normalTexels);
            quality.acne.slopeScaled =
                CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeScaled"), acneDefaults.slopeScaled);
            quality.acne.slopeMax =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeMax"), acneDefaults.slopeMax);
            interp->SetShadowMapQuality(quality);
        }
        interp->SetShadowMapParams(
            shadowMapOn,
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.CascadeCount"), SHADOW_MAP_DEFAULT_CASCADES),
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.Resolution"), SHADOW_MAP_DEFAULT_RESOLUTION),
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.ActorResolution"), SHADOW_MAP_DEFAULT_ACTOR_RESOLUTION),
            splits, lightDir,
            CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.BlendFraction"), SHADOW_MAP_DEFAULT_BLEND_FRACTION),
            CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.Strength"), SHADOW_MAP_DEFAULT_STRENGTH),
            CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.MinCasterSize"), SHADOW_MAP_DEFAULT_MIN_CASTER_SIZE),
            // Debug view selector. Passed through as an integer-valued float and not clamped here: the
            // shader owns the list, so a value it does not recognise falls through to normal shading rather
            // than being silently remapped to a view the user did not ask for.
            //
            // 1 and 2 show the system's OUTPUT (cascade footprints; which caster layer occludes). 3, 4, 5
            // and 7 show its INPUT -- the receiver normal, where that normal came from, the filter's raw
            // coverage, the cascade selection. That second group is the one to reach for when a shadow is
            // the wrong SHAPE rather than in the wrong place. The full list, with how to read them against
            // each other, is in the shader's PSMain.
            (f32)CVarGetInteger(CVAR_DEVELOPER_TOOLS("ShadowMap.ShowCascadeBounds"), 0), cascadeDivisors);
    }

    Fast::GfxRenderingAPI* rapi = GetRenderingApi();
    if (rapi == nullptr) {
        return;
    }
    // Developer diagnostic: when on, the renderer draws every toon-lit object as flat white (lit) /
    // black (shadow) so it is obvious which draws receive toon lighting (e.g. confirming whether large
    // water/lava surfaces are being relit and causing the ramp edge to flicker across them).
    f32 debugBands = CVarGetInteger(CVAR_DEVELOPER_TOOLS("ToonLighting.HighlightBands"), 0) ? 1.0f : 0.0f;
    rapi->SetToonRamp(
        CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ToonLighting.RampCenter"), kDefaultRampCenter),
        CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ToonLighting.RampSoftness"), kDefaultRampSoftness),
        CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ToonLighting.HighlightIntensity"), kDefaultHighlightIntensity),
        CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ToonLighting.ShadowIntensity"), kDefaultShadowIntensity), debugBands);
}

// ---------------------------------------------------------------------------------------------------
// Per-actor key-light state (eased "travel" between light sources)
// ---------------------------------------------------------------------------------------------------

// The key light animates smoothly toward its target, so the transition persists across frames. State
// is keyed by the actor pointer and evicted on the actor's destroy hook, so two live actors never
// collide and stale pointers never linger.
struct ToonKeyState {
    f32 dir[3];
    f32 col[3];
    f32 colVel[3];
    const LightInfo* localSources[TOON_LOCAL_LIGHT_MAX]{}; // compared only against this frame
    bool multipleLights = false;
    f32 shadowScale;    // actor-shadow size, eased 0..1 so it grows in / shrinks out instead of popping
    f32 shadowScaleVel; // SmoothDamp velocity for shadowScale
    // Cached floor raycast, for actors that never run a bg check (floorPoly stays NULL): a static
    // actor pays ONE raycast ever instead of one per frame; movers re-sample after ~4 units of drift.
    f32 floorY;      // cached raycast result (only meaningful when floorSampled)
    f32 floorPos[3]; // world position the raycast was sampled at
    u8 floorValid;   // the cached raycast hit a floor
    u8 floorSampled; // a raycast has been cached
};

static std::unordered_map<Actor*, ToonKeyState> sToonKeyStates;

// The eased state must not survive a disabled stretch (it would be stale on re-enable); cleared from
// the frame hook when both features are off. Prototyped above OnToonFrameUpdate.
static void ToonClearKeyStates() {
    if (!sToonKeyStates.empty()) {
        sToonKeyStates.clear();
    }
}

// Critically-damped smoothing (Unity-style SmoothDamp): eases a value toward a moving target with no
// overshoot, accelerating then decelerating, for a smooth toon-key "travel".
static f32 ToonSmoothDamp(f32 current, f32 target, f32* vel, f32 smoothTime, f32 dt) {
    if (smoothTime < 0.0001f) {
        smoothTime = 0.0001f;
    }
    f32 omega = 2.0f / smoothTime;
    f32 x = omega * dt;
    f32 expTerm = 1.0f / (1.0f + x + (0.48f * x * x) + (0.235f * x * x * x));
    f32 change = current - target;
    f32 temp = (*vel + (omega * change)) * dt;
    *vel = (*vel - (omega * temp)) * expTerm;
    return target + ((change + temp) * expTerm);
}

// Antipode-safe spherical interpolation of a unit direction by fraction t. Rotating along the sphere
// (never through the centre) keeps the key from snapping when the dominant light swings to the far
// side -- e.g. when a fairy's light switches off. The near-aligned fast path uses no trig, so the
// common per-frame case is cheap.
static void ToonSlerp(f32 from[3], f32 to[3], f32 t, f32 out[3]) {
    f32 dot = (from[0] * to[0]) + (from[1] * to[1]) + (from[2] * to[2]);
    f32 len;

    dot = (dot < -1.0f) ? -1.0f : ((dot > 1.0f) ? 1.0f : dot);

    if (dot > 0.9995f) {
        // Almost aligned: cheap linear step + renormalize (no trig).
        out[0] = from[0] + ((to[0] - from[0]) * t);
        out[1] = from[1] + ((to[1] - from[1]) * t);
        out[2] = from[2] + ((to[2] - from[2]) * t);
        len = sqrtf((out[0] * out[0]) + (out[1] * out[1]) + (out[2] * out[2]));
        if (len > 0.0001f) {
            out[0] /= len, out[1] /= len, out[2] /= len;
        }
        return;
    }

    if (dot < -0.9995f) {
        // Almost opposite: the great-circle path is ambiguous, so rotate around an arbitrary
        // perpendicular axis to reach the far side smoothly.
        f32 ref[3], perp[3], d, ang, c, s;

        if (fabsf(from[0]) < 0.9f) {
            ref[0] = 1.0f, ref[1] = 0.0f, ref[2] = 0.0f;
        } else {
            ref[0] = 0.0f, ref[1] = 1.0f, ref[2] = 0.0f;
        }
        d = (ref[0] * from[0]) + (ref[1] * from[1]) + (ref[2] * from[2]);
        perp[0] = ref[0] - (from[0] * d), perp[1] = ref[1] - (from[1] * d), perp[2] = ref[2] - (from[2] * d);
        len = sqrtf((perp[0] * perp[0]) + (perp[1] * perp[1]) + (perp[2] * perp[2]));
        if (len > 0.0001f) {
            perp[0] /= len, perp[1] /= len, perp[2] /= len;
        }
        ang = t * 3.14159265f;
        c = cosf(ang), s = sinf(ang);
        out[0] = (from[0] * c) + (perp[0] * s);
        out[1] = (from[1] * c) + (perp[1] * s);
        out[2] = (from[2] * c) + (perp[2] * s);
        return;
    }

    f32 theta = acosf(dot);
    f32 sinTheta = sinf(theta);
    f32 s0 = sinf((1.0f - t) * theta) / sinTheta;
    f32 s1 = sinf(t * theta) / sinTheta;
    out[0] = (s0 * from[0]) + (s1 * to[0]);
    out[1] = (s0 * from[1]) + (s1 * to[1]);
    out[2] = (s0 * from[2]) + (s1 * to[2]);
}

// ---------------------------------------------------------------------------------------------------
// Key-light selection
// ---------------------------------------------------------------------------------------------------

// Find the CLOSEST in-range point light (fairy, torch, bomb, ...) to the actor and, if any, fill the
// key direction (toward the light) + its colour. Brightness is intentionally ignored -- proximity
// alone decides -- so flickering torches are perfectly stable and the nearer of two always wins. A
// light is "in range" out to its radius × pointRange (raise pointRange to extend reach).
static bool ToonClosestPointLight(PlayState* play, Actor* actor, f32 pointRange, f32 dirOut[3], f32 colOut[3]) {
    // When the player opts Navi out, her two emitted lights are skipped by address (she blinks on/off
    // and orbits Link, so she'd constantly steal the key light). Resolved once per frame in
    // OnToonFrameUpdate -- identical for every actor.
    LightInfo* naviGlow = sNaviGlow;
    LightInfo* naviNoGlow = sNaviNoGlow;

    LightNode* node = play->lightCtx.listHead;
    f32 bestDistSq = -1.0f;
    // The winner, not the winner's outputs. The square root and the six divisions below used to run on
    // every light that improved on the one before it -- so walking a torchlit room from far to near paid
    // them again for each -- to produce values all but the last were about to overwrite. The comparison
    // that picks the winner only ever needed the SQUARED distance, so nothing else has to be computed
    // until the walk is over. Selection is unchanged: still the first light at the strict minimum.
    LightInfo* best = NULL;

    while (node != NULL) {
        LightInfo* info = node->info;

        if ((info != NULL) && (info->type != LIGHT_DIRECTIONAL) && (info != naviGlow) && (info != naviNoGlow)) {
            f32 dx = info->params.point.x - actor->world.pos.x;
            f32 dy = info->params.point.y - actor->world.pos.y;
            f32 dz = info->params.point.z - actor->world.pos.z;
            f32 radius = info->params.point.radius * pointRange;
            f32 distSq = (dx * dx) + (dy * dy) + (dz * dz);

            if ((radius > 0.0f) && (distSq > 0.0001f) && (distSq < (radius * radius)) &&
                ((bestDistSq < 0.0f) || (distSq < bestDistSq))) {
                bestDistSq = distSq;
                best = info;
            }
        }
        node = node->next;
    }

    if (best != NULL) {
        // Recomputed from the same light and the same actor position, neither of which the walk touched,
        // so these are the values the old code would have left behind on its last improving light.
        const f32 dx = best->params.point.x - actor->world.pos.x;
        const f32 dy = best->params.point.y - actor->world.pos.y;
        const f32 dz = best->params.point.z - actor->world.pos.z;
        const f32 dist = sqrtf(bestDistSq);
        dirOut[0] = dx / dist, dirOut[1] = dy / dist, dirOut[2] = dz / dist;
        colOut[0] = best->params.point.color[0] / 255.0f;
        colOut[1] = best->params.point.color[1] / 255.0f;
        colOut[2] = best->params.point.color[2] / 255.0f;
    }
    return bestDistSq >= 0.0f;
}

// Select at most four live local sources. The 20% retention bonus avoids swaps
// caused by torch flicker. Stored pointers are identity hints only, never dereferenced.
static void EmitToonLocalLights(PlayState* play, Actor* actor, ToonKeyState& state, bool enabled) {
    struct Candidate {
        LightInfo* info;
        float score;
        float attenuation;
        float distance2;
    } best[TOON_LOCAL_LIGHT_MAX]{};
    if (enabled) {
        for (LightNode* node = play->lightCtx.listHead; node != nullptr; node = node->next) {
            LightInfo* info = node->info;
            if (info == nullptr || info->type == LIGHT_DIRECTIONAL || info == sNaviGlow || info == sNaviNoGlow) {
                continue;
            }
            const auto& p = info->params.point;
            float dx = p.x - actor->world.pos.x, dy = p.y - actor->world.pos.y, dz = p.z - actor->world.pos.z;
            float distance2 = dx * dx + dy * dy + dz * dz;
            float attenuation = ToonLocalLightAttenuation(distance2, p.radius * sParams.pointRange);
            float score = attenuation * (0.2126f * p.color[0] + 0.7152f * p.color[1] + 0.0722f * p.color[2]);
            if (!(score > 0.0f)) {
                continue;
            }
            for (const LightInfo* previous : state.localSources) {
                if (previous == info) {
                    score *= 1.2f;
                    break;
                }
            }
            Candidate candidate{info, score, attenuation, distance2};
            for (int i = 0; i < TOON_LOCAL_LIGHT_MAX; ++i) {
                if (candidate.score > best[i].score) {
                    std::swap(candidate, best[i]);
                }
            }
        }
    }
    OPEN_DISPS(play->state.gfxCtx);
    gSPToonLocalLightsReset(POLY_OPA_DISP++, enabled);
    gSPToonLocalLightsReset(POLY_XLU_DISP++, enabled);
    for (int i = 0; i < TOON_LOCAL_LIGHT_MAX; ++i) {
        state.localSources[i] = best[i].info;
        if (best[i].info == nullptr) {
            continue;
        }
        const auto& p = best[i].info->params.point;
        float invDistance = best[i].distance2 > 0.0001f ? 1.0f / sqrtf(best[i].distance2) : 0.0f;
        // A source exactly at the object origin has no direction; choose up consistently.
        s8 dx = (s8)((p.x - actor->world.pos.x) * invDistance * 127.0f);
        s8 dy = invDistance > 0.0f ? (s8)((p.y - actor->world.pos.y) * invDistance * 127.0f) : 127;
        s8 dz = (s8)((p.z - actor->world.pos.z) * invDistance * 127.0f);
        float strength = best[i].attenuation * sParams.localIntensity;
        u8 r = (u8)(p.color[0] * strength), g = (u8)(p.color[1] * strength), b = (u8)(p.color[2] * strength);
        gSPToonLocalLight(POLY_OPA_DISP++, i, dx, dy, dz, r, g, b);
        gSPToonLocalLight(POLY_XLU_DISP++, i, dx, dy, dz, r, g, b);
    }
    CLOSE_DISPS(play->state.gfxCtx);
}

// Key light from the environment directionals: the sun or the moon, whichever is currently brighter
// (so it tracks day/night). Always fills dirOut/colOut.
static void ToonEnvKey(PlayState* play, f32 dirOut[3], f32 colOut[3]) {
    LightInfo* sun = &play->envCtx.dirLight1;
    LightInfo* moon = &play->envCtx.dirLight2;
    s32 sunLum = sun->params.dir.color[0] + sun->params.dir.color[1] + sun->params.dir.color[2];
    s32 moonLum = moon->params.dir.color[0] + moon->params.dir.color[1] + moon->params.dir.color[2];
    LightInfo* env = (moonLum > sunLum) ? moon : sun;
    f32 d0 = env->params.dir.x, d1 = env->params.dir.y, d2 = env->params.dir.z;
    f32 len = sqrtf((d0 * d0) + (d1 * d1) + (d2 * d2));

    if (len > 0.001f) {
        dirOut[0] = d0 / len, dirOut[1] = d1 / len, dirOut[2] = d2 / len;
        colOut[0] = env->params.dir.color[0] / 255.0f;
        colOut[1] = env->params.dir.color[1] / 255.0f;
        colOut[2] = env->params.dir.color[2] / 255.0f;
    }
}

// ---------------------------------------------------------------------------------------------------
// Debug visualizer (dev-tools only)
// ---------------------------------------------------------------------------------------------------

// A thin 4-sided spike along +Y (base at the origin, tip at y=1). Scaled/rotated, it becomes a "light
// ray" pointing from an actor toward a light.
static Vtx sToonRayVtx[5] = {
    VTX(-1, 0, -1, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF), // base
    VTX(1, 0, -1, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),  VTX(1, 0, 1, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(-1, 0, 1, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),  VTX(0, 1, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF), // tip
};

static Gfx sToonRayDL[] = {
    gsSPVertex(sToonRayVtx, 5, 0),
    gsSP2Triangles(0, 1, 4, 0, 1, 2, 4, 0),
    gsSP2Triangles(2, 3, 4, 0, 3, 0, 4, 0),
    gsSP2Triangles(0, 2, 1, 0, 0, 3, 2, 0),
    gsSPEndDisplayList(),
};

// A flat 12-segment ring (annulus) in the XZ plane at base radius 100 (interleaved inner/outer
// verts). Scaled and centred at a point light, it shows that light's effective range so the Point
// Light Range slider's reach is visible.
static Vtx sToonRingVtx[24] = {
    VTX(97, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),    VTX(103, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(84, 0, 48, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),   VTX(89, 0, 52, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(48, 0, 84, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),   VTX(52, 0, 89, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(0, 0, 97, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),    VTX(0, 0, 103, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(-48, 0, 84, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),  VTX(-52, 0, 89, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(-84, 0, 48, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),  VTX(-89, 0, 52, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(-97, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),   VTX(-103, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(-84, 0, -48, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF), VTX(-89, 0, -52, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(-48, 0, -84, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF), VTX(-52, 0, -89, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(0, 0, -97, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),   VTX(0, 0, -103, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(48, 0, -84, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),  VTX(52, 0, -89, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
    VTX(84, 0, -48, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),  VTX(89, 0, -52, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF),
};

static Gfx sToonRingDL[] = {
    gsSPVertex(sToonRingVtx, 24, 0),
    gsSP2Triangles(0, 1, 3, 0, 0, 3, 2, 0),
    gsSP2Triangles(2, 3, 5, 0, 2, 5, 4, 0),
    gsSP2Triangles(4, 5, 7, 0, 4, 7, 6, 0),
    gsSP2Triangles(6, 7, 9, 0, 6, 9, 8, 0),
    gsSP2Triangles(8, 9, 11, 0, 8, 11, 10, 0),
    gsSP2Triangles(10, 11, 13, 0, 10, 13, 12, 0),
    gsSP2Triangles(12, 13, 15, 0, 12, 15, 14, 0),
    gsSP2Triangles(14, 15, 17, 0, 14, 17, 16, 0),
    gsSP2Triangles(16, 17, 19, 0, 16, 19, 18, 0),
    gsSP2Triangles(18, 19, 21, 0, 18, 21, 20, 0),
    gsSP2Triangles(20, 21, 23, 0, 20, 23, 22, 0),
    gsSP2Triangles(22, 23, 1, 0, 22, 1, 0, 0),
    gsSPEndDisplayList(),
};

// Draw one debug "light ray": a flat-coloured spike from `base` pointing along the unit direction
// `dir`, `length` long and `thickness` wide. Drawn translucent and depth-test-free so every ray is
// visible (debug only -- gated by the caller).
static void DrawDebugRay(PlayState* play, Vec3f* base, f32 dir[3], u8 r, u8 g, u8 b, f32 length, f32 thickness) {
    Vec3f axis;
    f32 horiz;

    OPEN_DISPS(play->state.gfxCtx);

    gDPPipeSync(POLY_XLU_DISP++);
    gSPClearGeometryMode(POLY_XLU_DISP++, G_LIGHTING | G_CULL_BACK | G_CULL_FRONT);
    gDPSetCombineLERP(POLY_XLU_DISP++, 0, 0, 0, PRIMITIVE, 0, 0, 0, PRIMITIVE, 0, 0, 0, PRIMITIVE, 0, 0, 0, PRIMITIVE);
    gDPSetRenderMode(POLY_XLU_DISP++, G_RM_AA_XLU_SURF, G_RM_AA_XLU_SURF2);
    gDPSetPrimColor(POLY_XLU_DISP++, 0, 0, r, g, b, 200);

    // Orient the spike's +Y axis to `dir` (rotate +Y onto dir about their perpendicular).
    Matrix_Translate(base->x, base->y, base->z, MTXMODE_NEW);
    horiz = sqrtf((dir[0] * dir[0]) + (dir[2] * dir[2]));
    if (horiz > 0.001f) {
        axis.x = dir[2] / horiz; // normalized cross((0,1,0), dir) = (dir.z, 0, -dir.x)
        axis.y = 0.0f;
        axis.z = -dir[0] / horiz;
        Matrix_RotateAxis(Math_FAtan2F(horiz, dir[1]), &axis, MTXMODE_APPLY);
    } else if (dir[1] < 0.0f) {
        axis.x = 1.0f, axis.y = 0.0f, axis.z = 0.0f; // pointing straight down
        Matrix_RotateAxis(M_PI, &axis, MTXMODE_APPLY);
    }
    Matrix_Scale(thickness, length, thickness, MTXMODE_APPLY);

    gSPMatrix(POLY_XLU_DISP++, MATRIX_NEWMTX(play->state.gfxCtx), G_MTX_MODELVIEW | G_MTX_LOAD);
    gSPDisplayList(POLY_XLU_DISP++, sToonRayDL);

    CLOSE_DISPS(play->state.gfxCtx);
}

// Draw a flat debug ring of the given world `radius` centred at `center`, used to show a point
// light's effective range. Depth-test-free so it's always visible (debug only).
static void DrawDebugRing(PlayState* play, Vec3f* center, f32 radius, u8 r, u8 g, u8 b, u8 a) {
    OPEN_DISPS(play->state.gfxCtx);

    gDPPipeSync(POLY_XLU_DISP++);
    gSPClearGeometryMode(POLY_XLU_DISP++, G_LIGHTING | G_CULL_BACK | G_CULL_FRONT);
    gDPSetCombineLERP(POLY_XLU_DISP++, 0, 0, 0, PRIMITIVE, 0, 0, 0, PRIMITIVE, 0, 0, 0, PRIMITIVE, 0, 0, 0, PRIMITIVE);
    gDPSetRenderMode(POLY_XLU_DISP++, G_RM_AA_XLU_SURF, G_RM_AA_XLU_SURF2);
    gDPSetPrimColor(POLY_XLU_DISP++, 0, 0, r, g, b, a);

    Matrix_Translate(center->x, center->y, center->z, MTXMODE_NEW);
    Matrix_Scale(radius * 0.01f, 1.0f, radius * 0.01f, MTXMODE_APPLY); // ring geometry is at base radius 100
    gSPMatrix(POLY_XLU_DISP++, MATRIX_NEWMTX(play->state.gfxCtx), G_MTX_MODELVIEW | G_MTX_LOAD);
    gSPDisplayList(POLY_XLU_DISP++, sToonRingDL);

    CLOSE_DISPS(play->state.gfxCtx);
}

// The candidate-light rays, range rings, and the chosen-key needle (see the per-control help in the
// dev-tools viewer toggle). Gated by the caller; reads the same lights the selection does.
static void DrawDebugOverlay(PlayState* play, Actor* actor, f32 pointRange, f32 chosenDir[3]) {
    Player* player = GET_PLAYER(play);
    Vec3f base = { actor->world.pos.x, actor->world.pos.y + 30.0f, actor->world.pos.z };
    f32 cdir[3];
    LightNode* dn;
    // Draw the per-light range rings just once (on the player's pass) so they aren't redrawn by every
    // actor in the scene.
    s32 isPlayer = (player != NULL) && (actor == &player->actor);

    for (s32 i = 0; i < 2; i++) {
        LightInfo* env = (i == 0) ? &play->envCtx.dirLight1 : &play->envCtx.dirLight2;
        f32 e0 = env->params.dir.x, e1 = env->params.dir.y, e2 = env->params.dir.z;
        f32 elen = sqrtf((e0 * e0) + (e1 * e1) + (e2 * e2));
        f32 elum = (env->params.dir.color[0] + env->params.dir.color[1] + env->params.dir.color[2]) / (3.0f * 255.0f);

        if (elen > 0.001f) {
            cdir[0] = e0 / elen, cdir[1] = e1 / elen, cdir[2] = e2 / elen;
            DrawDebugRay(play, &base, cdir, env->params.dir.color[0], env->params.dir.color[1],
                         env->params.dir.color[2], 10.0f + (elum > 1.0f ? 1.0f : elum) * 15.0f, 1.2f);
        }
    }

    dn = play->lightCtx.listHead;
    while (dn != NULL) {
        LightInfo* info = dn->info;

        if ((info != NULL) && (info->type != LIGHT_DIRECTIONAL)) {
            f32 dx = info->params.point.x - actor->world.pos.x;
            f32 dy = info->params.point.y - actor->world.pos.y;
            f32 dz = info->params.point.z - actor->world.pos.z;
            f32 radius = info->params.point.radius * pointRange;
            f32 dsq = (dx * dx) + (dy * dy) + (dz * dz);

            // Cyan range ring at each point light (once, on the player pass), for ALL point lights
            // regardless of distance, so the Point Light Range slider's reach is visible.
            if (isPlayer && (radius > 0.0f)) {
                Vec3f lpos = { info->params.point.x, info->params.point.y, info->params.point.z };
                DrawDebugRing(play, &lpos, radius, 0, 255, 255, 110);
            }

            if ((radius > 0.0f) && (dsq < (radius * radius))) {
                f32 dist = sqrtf(dsq);
                f32 scale = 1.0f - ((dist / radius) * (dist / radius));
                f32 att = 0.5f + (0.5f * scale); // distance falloff, just for the visual length
                f32 plum = ((info->params.point.color[0] + info->params.point.color[1] + info->params.point.color[2]) /
                            (3.0f * 255.0f)) *
                           att;

                if (dist > 0.001f) {
                    cdir[0] = dx / dist, cdir[1] = dy / dist, cdir[2] = dz / dist;
                    DrawDebugRay(play, &base, cdir, (u8)(info->params.point.color[0] * att),
                                 (u8)(info->params.point.color[1] * att), (u8)(info->params.point.color[2] * att),
                                 10.0f + (plum > 2.0f ? 2.0f : plum) * 12.5f, 1.2f);
                }
            }
        }
        dn = dn->next;
    }

    // The chosen key -- a thin MAGENTA needle down the centre of the chosen light's cone, so the
    // direction is marked while the light's own colour is still visible in its (wider) cone.
    DrawDebugRay(play, &base, chosenDir, 255, 0, 255, 35.0f, 0.3f);
}

// ---------------------------------------------------------------------------------------------------
// Per-actor draw: choose, ease, and emit this actor's key light
// ---------------------------------------------------------------------------------------------------

// Choose this actor's key (closest in-range point light, else the sun/moon), ease it toward that target
// with an antipode-safe slerp, and emit it via gSPToonKey. Runs inside Actor_Draw's display-list scope
// (via the OnActorDraw hook), so the key precedes the actor's geometry.
static void HandleActorDraw(void* actorPtr) {
    PlayState* play = gPlayState;
    Actor* actor = (Actor*)actorPtr;
    if (play == NULL || actor == NULL) {
        return;
    }

    bool celEnabled = sParams.cel;
    // The caster marker (gSPToonShadowArm) is what tells the renderer "this object casts a shadow", and BOTH
    // non-vanilla systems need it: the stencil volumes build a silhouette from it, and the shadow map
    // captures the same geometry in world space for its depth pass. Gating it on the stencil mode alone left
    // shadow-map mode with no casters at all, so its cascades came out empty and nothing was ever shadowed.
    // The renderer decides which technique consumes the capture; here we only mark the casters.
    bool shadowsEnabled = sParams.shadows || ToonLighting_ShadowMapEnabled() != 0;
    if (!celEnabled && !shadowsEnabled) {
        return; // hooks stay registered so console toggles work; the per-frame snapshot gates the work
    }

    // Close the previous actor's scenery bracket, if it had one. This runs before the actor draws and
    // therefore before any of its own geometry is appended, which is what makes the bracket cover one actor
    // exactly. It has to sit ahead of the early returns below, or an excluded actor drawing right after a
    // tree would have its geometry counted as part of that tree's.
    //
    // Both buffers, and not merged into the disarm the excluded-actor path emits a few lines down: that one
    // is gSPToonShadow(0), which clears the ACTOR arming, and the two brackets are separate flags on the
    // renderer. Closing one has never closed the other.
    if (sSceneryCasterArmed) {
        OPEN_DISPS(play->state.gfxCtx);
        gSPShadowMapSceneryCasterEnd(POLY_OPA_DISP++);
        gSPShadowMapSceneryCasterEnd(POLY_XLU_DISP++);
        CLOSE_DISPS(play->state.gfxCtx);
        sSceneryCasterArmed = false;
    }

    // Blacklist (doors/trees/water): excluded actors get neither cel relight nor a shadow. When cel shading
    // is on, flip its bracket OFF around them (deduped via sToonEnabled) so they keep vanilla lighting; the
    // next normal actor flips it back ON. The bracket only exists while cel shading is on, so skip the flip
    // otherwise (a shadow alone never relights). Own disp scope so excluded actors return cleanly without an
    // unbalanced CLOSE_DISPS.
    bool wantToon = !ToonActorExcluded(actor);
    if (celEnabled && wantToon != sToonEnabled) {
        OPEN_DISPS(play->state.gfxCtx);
        gSPToon(POLY_OPA_DISP++, wantToon);
        gSPToon(POLY_XLU_DISP++, wantToon);
        CLOSE_DISPS(play->state.gfxCtx);
        sToonEnabled = wantToon;
        // Every gSPToon edge invalidates the renderer's key (it expects a fresh gSPToonKey per toon-on),
        // so drop the dedup state too -- otherwise the next actor with an unchanged key would skip its
        // emit and render with the renderer's neutral fallback key instead of the real one.
        sHaveLastKey = false;
    }
    // Being excluded from the cel relight does not mean the thing does not block light: a tree is on that
    // list because its canopy should keep vanilla shading, and a tree is also the single most obvious
    // shadow caster in the game. Shadow-map mode therefore carries these actors through to the arming
    // block below instead of returning here.
    //
    // The stencil volumes still stop: that technique casts the silhouette along the actor's OWN key light,
    // and an actor excluded from the relight never gets one. Shadow maps have no such dependency -- their
    // light direction is frame-global (see ToonEnvKey in the pusher).
    const bool relightExcluded = !wantToon;
    const bool castWithoutRelight = relightExcluded && shadowsEnabled && ToonLighting_ShadowMapEnabled() != 0;
    if (relightExcluded && !castWithoutRelight) {
        // Excluded actors must still mark the per-object shadow boundary. With cel shading on, the
        // bracket edge above flushes+disarms the capture; with cel OFF there is no edge, and without
        // this disarm the PREVIOUS actor's silhouette would keep accumulating this actor's lit
        // geometry (a door or tree merging into a nearby NPC's shadow).
        if (shadowsEnabled) {
            OPEN_DISPS(play->state.gfxCtx);
            gSPToonShadow(POLY_OPA_DISP++, 0, 0, 0, 0.0f);
            CLOSE_DISPS(play->state.gfxCtx);
        }
        return;
    }

    f32 targetDir[3] = { 0.0f, 1.0f, 0.0f }; // default: lit from above
    f32 targetCol[3] = { 1.0f, 1.0f, 1.0f };
    // How far a point light reaches (× its radius) for selection, and how long the eased travel takes.
    f32 pointRange = sParams.pointRange;
    f32 transitionTime = sParams.transitionTime;

    OPEN_DISPS(play->state.gfxCtx);

    // Multi-light keeps the environment key. Legacy mode lets the closest local source replace it.
    const bool multipleLights = celEnabled && wantToon && sParams.multipleLights;
    if (multipleLights || !ToonClosestPointLight(play, actor, pointRange, targetDir, targetCol)) {
        ToonEnvKey(play, targetDir, targetCol);
    }

    // Animate the key toward the chosen light with an eased "travel" (per-actor persistent state).
    auto [it, isNew] = sToonKeyStates.try_emplace(actor);
    ToonKeyState& st = it->second;
    if (isNew || st.multipleLights != multipleLights) {
        st.colVel[0] = st.colVel[1] = st.colVel[2] = 0.0f;
        st.dir[0] = targetDir[0], st.dir[1] = targetDir[1], st.dir[2] = targetDir[2];
        st.col[0] = targetCol[0], st.col[1] = targetCol[1], st.col[2] = targetCol[2];
        if (isNew) {
            st.shadowScale = 0.0f, st.shadowScaleVel = 0.0f; // grows in on first appearance
            st.floorSampled = 0, st.floorValid = 0;
        }
    } else {
        // Eased travel using the frame-constant dt/alpha computed in OnToonFrameUpdate (frame
        // interpolation replays this draw without re-running it, so they can't vary per actor anyway).
        f32 newDir[3];

        ToonSlerp(st.dir, targetDir, sToonKeyAlpha, newDir);
        st.dir[0] = newDir[0], st.dir[1] = newDir[1], st.dir[2] = newDir[2];
        for (s32 i = 0; i < 3; i++) {
            st.col[i] = ToonSmoothDamp(st.col[i], targetCol[i], &st.colVel[i], transitionTime, sToonKeyDt);
        }
    }

    st.multipleLights = multipleLights;
    if (!castWithoutRelight) {
        EmitToonLocalLights(play, actor, st, multipleLights);
    }

    {
        s8 dx = (s8)(st.dir[0] * 127.0f);
        s8 dy = (s8)(st.dir[1] * 127.0f);
        s8 dz = (s8)(st.dir[2] * 127.0f);
        u8 r = (u8)((st.col[0] < 0.0f ? 0.0f : (st.col[0] > 1.0f ? 1.0f : st.col[0])) * 255.0f);
        u8 g = (u8)((st.col[1] < 0.0f ? 0.0f : (st.col[1] > 1.0f ? 1.0f : st.col[1])) * 255.0f);
        u8 b = (u8)((st.col[2] < 0.0f ? 0.0f : (st.col[2] > 1.0f ? 1.0f : st.col[2])) * 255.0f);

        // Emit only when the quantized key changed (see sHaveLastKey above). Both display lists get the
        // key together so they stay in lockstep.
        bool keyChanged = !sHaveLastKey || dx != sLastKeyDir[0] || dy != sLastKeyDir[1] || dz != sLastKeyDir[2] ||
                          r != sLastKeyCol[0] || g != sLastKeyCol[1] || b != sLastKeyCol[2];
        // Never for a cast-only actor: its toon bracket is OFF, and emitting a key inside an off bracket
        // would leave the renderer holding this actor's key for whatever relit object comes next.
        if (keyChanged && !castWithoutRelight) {
            gSPToonKey(POLY_OPA_DISP++, dx, dy, dz, r, g, b);
            gSPToonKey(POLY_XLU_DISP++, dx, dy, dz, r, g, b);
            sLastKeyDir[0] = dx, sLastKeyDir[1] = dy, sLastKeyDir[2] = dz;
            sLastKeyCol[0] = r, sLastKeyCol[1] = g, sLastKeyCol[2] = b;
            sHaveLastKey = true;
        }
    }

    // Actor shadow: arm this actor's drop shadow. The renderer builds a stencil volume from the actor's
    // captured silhouette and casts it along the key just snapshotted (gSPToonKey above) onto the real ground,
    // so it conforms to slopes and always agrees with the cel shading. POLY_OPA only, so translucent effects
    // don't cast. Emitted for every non-excluded actor when on (zero normal disarms it) so the per-object
    // boundary is always marked and the previous actor's capture can't leak into this one.
    if (shadowsEnabled) {
        // The shadow shows when the actor is on/near the ground, within the render-distance cull, and NOT on a
        // wall -- climbing a ladder/vine or climbing/hanging off a ledge, where it's flat against a vertical
        // surface and the ground shadow's slab would cut into the wall and leave broken lines. Rather than pop
        // on/off, the SIZE eases 0..1 (like Navi's light) so it grows in / shrinks to nothing. The eased scale
        // rides in planeD; the renderer scales the footprint by it (it ignores the floor plane otherwise), and
        // any nonzero normal simply arms the pass. A zero normal fully disarms it (no capture/projection/draw).
        f32 maxDist = sParams.maxDist;
        bool onWall = false;
        if (actor->id == ACTOR_PLAYER) {
            Player* player = (Player*)actor;
            onWall = (player->stateFlags1 & (PLAYER_STATE1_HANGING_OFF_LEDGE | PLAYER_STATE1_CLIMBING_LEDGE |
                                             PLAYER_STATE1_CLIMBING_LADDER)) != 0;
        }
        bool hasFloor = false;
        f32 floorHeight = actor->floorHeight;
        // The lower bound matters with the extended-culling enhancements: they draw actors BEHIND the
        // camera (negative projected z), which would otherwise pay full capture + volume cost for a
        // shadow that is never visible.
        // Shadow maps need a different reach test than the stencil volumes do.
        //
        // projectedPos.z is view-space depth, so it changes for every actor the moment the camera turns.
        // For the stencil volumes that was fine -- those are composited in screen space, so a caster out of
        // view had no visible shadow anyway. A depth map is the opposite: something beside or behind the
        // camera casts into the view perfectly well, and culling it by view depth made its shadow blink out
        // as soon as the camera moved. The "behind the camera" bound (-100) did the same thing.
        //
        // Radial distance from the camera is rotation-invariant, which is the property this needs, and the
        // reach is the furthest cascade -- past that there is no map left to record the caster in.
        bool withinShadowReach;
        if (ToonLighting_ShadowMapEnabled()) {
            const f32 px = actor->projectedPos.x, py = actor->projectedPos.y, pz = actor->projectedPos.z;
            const f32 radial = sqrtf((px * px) + (py * py) + (pz * pz));
            withinShadowReach = radial < sParams.shadowMapReach;
        } else {
            withinShadowReach = actor->projectedPos.z < maxDist && actor->projectedPos.z > -100.0f;
        }
        if (!ToonShadowExcluded(actor) && withinShadowReach) {
            // Floor reference is the gate + a "near the ground" sanity check, and the feet-clamp Y for
            // deep-rooted actors (the renderer otherwise builds the volume from the captured feet, not this
            // plane). Most actors expose actor->floorPoly from their bg check; a few (e.g. the Courtyard Guards,
            // En_Heishi1) never run one, so floorPoly stays null and the shadow would never arm. Fall back to a
            // downward raycast for those -- cached in the eased state (see ToonKeyState) so stationary
            // actors don't re-walk the static collision every frame.
            bool haveFloor = (actor->floorPoly != NULL);
            if (!haveFloor) {
                f32 mdx = actor->world.pos.x - st.floorPos[0];
                f32 mdy = actor->world.pos.y - st.floorPos[1];
                f32 mdz = actor->world.pos.z - st.floorPos[2];
                if (!st.floorSampled || ((mdx * mdx) + (mdy * mdy) + (mdz * mdz)) > 16.0f) {
                    Vec3f rayFrom = { actor->world.pos.x, actor->world.pos.y + 1.0f, actor->world.pos.z };
                    CollisionPoly* poly = NULL;
                    st.floorY = BgCheck_EntityRaycastFloor2(play, &play->colCtx, &poly, &rayFrom);
                    st.floorValid = (poly != NULL);
                    st.floorSampled = 1;
                    st.floorPos[0] = actor->world.pos.x;
                    st.floorPos[1] = actor->world.pos.y;
                    st.floorPos[2] = actor->world.pos.z;
                }
                haveFloor = st.floorValid;
                if (haveFloor) {
                    floorHeight = st.floorY;
                }
            }
            if (haveFloor) {
                f32 distToFloor = actor->world.pos.y - floorHeight;
                hasFloor = (distToFloor > -50.0f) && (distToFloor < 1500.0f);
            }
        }
        st.shadowScale = ToonSmoothDamp(st.shadowScale, (hasFloor && !onWall) ? 1.0f : 0.0f, &st.shadowScaleVel,
                                        kShadowFadeTime, sToonKeyDt);
        if (st.shadowScale > 0.01f) {
            // Deep-rooted models (signposts) bury their geometry below the floor, which would sink the shadow
            // slab underground; pass the floor Y so the renderer lifts the slab's feet up to it. Everyone else
            // passes TOON_SHADOW_NO_CLAMP and keeps the captured feet.
            // A scenery actor is bracketed as scenery instead of armed as an actor, and in BOTH buffers,
            // because a tree is drawn into both -- trunk to POLY_OPA, canopy to POLY_XLU. Instead of, not as
            // well as: the actor arming would put the same geometry in the caster layer characters skip, and
            // whichever list a triangle lands in first is the one it stays in, so arming both would put the
            // tree back where its shadow cannot reach the player.
            //
            // Inside this branch rather than beside the whitelist test, so it inherits every gate the arming
            // passes through -- shadow reach, floor check, the eased fade. A tree past the last cascade
            // costs nothing in either stream, and its canopy is captured exactly when its trunk is.
            if (ToonLighting_ShadowMapEnabled() && ToonShadowSceneryCaster(actor)) {
                // Disarm the actor capture first, and explicitly. Not arming is not the same as being
                // disarmed: the marker is a running state in the stream, so without this the tree inherits
                // whichever actor drew before it, its geometry satisfies the actor branch in the renderer,
                // and it lands back in the layer this whole bracket exists to keep it out of. It also closes
                // that actor's object boundary, which is what the marker means anyway.
                gSPToonShadow(POLY_OPA_DISP++, 0, 0, 0, 0.0f);
                gSPShadowMapSceneryCasterBegin(POLY_OPA_DISP++);
                gSPShadowMapSceneryCasterBegin(POLY_XLU_DISP++);
                sSceneryCasterArmed = true;
                if (sParams.shadowMapCensus) {
                    sCensusAccum[actor->id]++; // see ToonShadowCensusPublish
                }
            } else {
                // Deep-rooted models (signposts) bury their geometry below the floor, which would sink the
                // shadow slab underground; pass the floor Y so the renderer lifts the slab's feet up to it.
                // Everyone else passes TOON_SHADOW_NO_CLAMP and keeps the captured feet.
                f32 clampY = floorHeight < -32767.0f ? -32767.0f : (floorHeight > 32767.0f ? 32767.0f : floorHeight);
                s16 feetClamp = ToonShadowDeepRooted(actor) ? (s16)clampY : (s16)TOON_SHADOW_NO_CLAMP;
                gSPToonShadowArm(POLY_OPA_DISP++, feetClamp, st.shadowScale); // planeD = eased size scale
            }
        } else {
            gSPToonShadow(POLY_OPA_DISP++, 0, 0, 0, 0.0f); // fully off
        }
    } else {
        // Shadows off (cel still on): keep the eased size at zero so re-enabling grows the shadow in
        // instead of popping it at whatever size it froze at.
        st.shadowScale = 0.0f;
        st.shadowScaleVel = 0.0f;
    }

    if (sParams.showDebug) {
        DrawDebugOverlay(play, actor, pointRange, st.dir);
    }

    CLOSE_DISPS(play->state.gfxCtx);
}

// Drop a destroyed actor's eased state so its slot can't be reused stale and the map can't grow
// unbounded over a session.
static void HandleActorDestroy(void* actorPtr) {
    sToonKeyStates.erase((Actor*)actorPtr);
}

// Close the last actor's translucent-pass caster bracket. Nothing draws after it inside the loop to do the
// job, and everything that draws after the loop -- effects, particles, water, the lens pass -- is in the same
// buffer, so leaving it open would feed all of it to the cutout path as though it were foliage.
extern "C" void ToonLighting_ShadowMapSceneryCasterClose(GraphicsContext* gfxCtx) {
    if (!sSceneryCasterArmed) {
        return;
    }
    OPEN_DISPS(gfxCtx);
    gSPShadowMapSceneryCasterEnd(POLY_OPA_DISP++);
    gSPShadowMapSceneryCasterEnd(POLY_XLU_DISP++);
    CLOSE_DISPS(gfxCtx);
    sSceneryCasterArmed = false;
}

// Lens-of-Truth actors draw through Actor_Draw AFTER the main actor bracket has closed, so without
// these the hook's bracket tracking (sToonEnabled) desyncs from the stream: lens actors miss their cel
// relight, an excluded lens actor can leave an unmatched bracket-ON leaking into the next frame, and
// the last lens actor's shadow capture stays armed into the XLU stream. Re-open the bracket around the
// lens pass (keeping the tracker in step) and mark the final object boundary on the way out.
extern "C" void ToonLighting_LensBracketBegin(GraphicsContext* gfxCtx) {
    if (sParams.cel) {
        OPEN_DISPS(gfxCtx);
        gSPToon(POLY_OPA_DISP++, true);
        gSPToon(POLY_XLU_DISP++, true);
        CLOSE_DISPS(gfxCtx);
        sToonEnabled = true;
        sHaveLastKey = false; // every bracket edge invalidates the renderer's key (see HandleActorDraw)
    }
}

extern "C" void ToonLighting_LensBracketEnd(GraphicsContext* gfxCtx) {
    if (!sParams.cel && !sParams.shadows) {
        return;
    }
    OPEN_DISPS(gfxCtx);
    if (sParams.cel) {
        gSPToon(POLY_OPA_DISP++, false);
        gSPToon(POLY_XLU_DISP++, false);
        sToonEnabled = false;
    }
    if (sParams.shadows && !sParams.cel) {
        // No closing bracket edge exists with cel off, so disarm the capture explicitly (the edge
        // above already does it when cel is on).
        gSPToonShadow(POLY_OPA_DISP++, 0, 0, 0, 0.0f);
    }
    CLOSE_DISPS(gfxCtx);
}

// The actor-shadow volumes are flushed by gSPToonShadowFlush emitted directly in the game's actor draw loop
// (soh/src/code/z_actor.c, func_800315AC) -- after the room and the walkable-floor receiver pre-pass, before
// the remaining actors. That placement is why it lives in the draw loop rather than a hook here: it has to sit
// between the two actor passes. The volumes are the previous frame's captures, so the shadow lags one frame
// (imperceptible for a ground shadow). The receiver whitelist it consults is ToonShadowReceiver above, exposed
// to C via ToonLighting_IsShadowReceiver.

// Carry a pre-mode-selector config forward: the feature used to be a plain on/off under
// Graphics.WorldShadows.Enabled, which is now one value of Graphics.WorldShadows.Mode. Runs once per
// launch and only when the old key is still present, so it cannot fight a later menu change (which
// writes Mode and never resurrects Enabled).
static void MigrateLegacyShadowToggle() {
    const char* kLegacyEnabled = CVAR_ENHANCEMENT("Graphics.WorldShadows.Enabled");
    if (CVarGetInteger(kLegacyEnabled, -1) < 0) {
        return; // never set, or already migrated
    }
    if (CVarGetInteger(kLegacyEnabled, 0) != 0) {
        CVarSetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode"), SHADOW_MODE_ACTOR);
    }
    CVarClear(kLegacyEnabled);
}

void RegisterToonLighting() {
    MigrateLegacyShadowToggle();
    // Registered unconditionally: the per-frame CVar snapshot (RefreshFrameParams) gates all the work,
    // which is what lets a console `set` of either Enabled CVar take effect without this re-running --
    // the game-code guard (ToonLighting_FeaturesActive) keeps the hook dispatch itself out of the
    // per-actor path when both features are off.
    COND_HOOK(OnGameFrameUpdate, true, OnToonFrameUpdate);
    COND_HOOK(OnActorDraw, true, HandleActorDraw);
    COND_HOOK(OnActorDestroy, true, HandleActorDestroy);
    RefreshFrameParams();
    // Drop the key-dedup state so the first actor after a (re-)enable always emits, before the
    // end-of-frame OnToonFrameUpdate reset has had a chance to run.
    sHaveLastKey = false;
    sToonEnabled = true;
    if (!sParams.cel && !sParams.shadows) {
        sToonKeyStates.clear();
    }
}

static RegisterShipInitFunc initFunc(RegisterToonLighting, { CVAR_ENHANCEMENT("Graphics.ToonLighting.Enabled"),
                                                             CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode") });
