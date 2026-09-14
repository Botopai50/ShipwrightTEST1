#include "SohMenu.h"
#include <cstdlib> // std::abs for the resolution snaps below
#include <cmath>   // std::fabs for the profile match test
#include "SohGui.hpp"
#include "soh/OTRGlobals.h"
#include "UIWidgets.hpp"
#include "soh/Enhancements/Graphics/ToonLighting.h"
#include <nlohmann/json.hpp> // the shadow capture's game_context, written by the button below

// Declared here rather than by including variables.h, which is the game's whole C symbol surface and pulls
// z64.h with it. The capture's game_context needs exactly these four, and other menu files declare
// gPlayState the same way. The spellings match variables.h exactly, so a change there fails to link here
// rather than diverging quietly.
extern PlayState* gPlayState;
extern "C" {
extern SaveContext gSaveContext;
extern const char gBuildVersion[];
extern const char gGitBranch[];
extern const char gGitCommitHash[];
}

namespace SohGui {

extern std::shared_ptr<SohMenu> mSohMenu;
using namespace UIWidgets;

// "Sombras" -- every shadow setting in the game, in one place.
//
// It used to be three: the mode selector and the cascade sliders under Wind Waker Style > Actor Shadows,
// the edge techniques under their own "Qualidade das Sombras" tab, and the bias controls under "Correção
// de Acne". Nothing about that split followed from the code -- a person changing "how sharp is the shadow"
// had to know which of three tabs owned that particular knob -- so the three are merged here and the two
// old tabs are gone.
//
// The shape of this menu is a PROFILE plus an ADVANCED switch. With the switch off you get the system
// selector, a profile, and the one taste knob (how dark). That is the whole surface, and it is enough to
// play with. With it on, every individual control appears in its own section, and the profile drops to
// "Personalizado" the moment any of them stops matching -- see ShadowProfileMatch below.
//
// Defaults are written out rather than pulled from fast/shadow_map.h, matching the convention the three
// original files used, and each one names the constant it must be kept in step with.

// Keyed by ShadowMode (see ToonLighting.h) -- the three shadow systems are mutually exclusive.
static const std::map<int32_t, const char*> shadowModeLabels = {
    { SHADOW_MODE_VANILLA, "Vanilla" },
    { SHADOW_MODE_ACTOR, "Actor Shadows" },
    { SHADOW_MODE_SHADOW_MAP, "Shadow Map" },
};

// Keyed by SHADOW_MAP_LAYOUT_*.
static const std::map<int32_t, const char*> shadowLayoutLabels = {
    { 0, "Cascatas" }, // SHADOW_MAP_LAYOUT_CASCADE
    { 1, "Clipmap" },  // SHADOW_MAP_LAYOUT_CLIPMAP
};

// Keyed by SHADOW_MAP_LADDER_*.
static const std::map<int32_t, const char*> shadowLadderModeLabels = {
    { 0, "Manual (sliders)" }, // SHADOW_MAP_LADDER_MANUAL
    { 1, "Automática" },       // SHADOW_MAP_LADDER_PRACTICAL
};

// The profile slots. "Personalizado" is not a preset anyone can pick INTO -- it is what the selector
// reports when the current settings match none of the four, which is how a hand-tuned configuration
// survives being looked at.
#define SHADOW_PROFILE_LOW 0
#define SHADOW_PROFILE_MEDIUM 1
#define SHADOW_PROFILE_HIGH 2
#define SHADOW_PROFILE_ULTRA 3
#define SHADOW_PROFILE_CUSTOM 4

static const std::map<int32_t, const char*> shadowProfileLabels = {
    { SHADOW_PROFILE_LOW, "Baixo" },   { SHADOW_PROFILE_MEDIUM, "Médio" },         { SHADOW_PROFILE_HIGH, "Alto" },
    { SHADOW_PROFILE_ULTRA, "Ultra" }, { SHADOW_PROFILE_CUSTOM, "Personalizado" },
};

// One profile, as the values it writes. Everything that changes how a shadow LOOKS is in here -- the map
// sizes, the cascade ladder, the refresh rates, the edge techniques and the acne bias -- because a profile
// that only moved half of them would be a profile that cannot honestly be called "Alto".
//
// Intensity is deliberately NOT here. It is not a quality setting: it is how dark you want your shadows,
// and no preset has any business overwriting that.
struct ShadowProfile {
    int32_t resolution;
    int32_t actorResolution;
    int32_t cascadeCount;
    int32_t divisor0, divisor1, divisor2;
    int32_t staticCache;
    int32_t analyticEdge;
    float analyticEdgeWidth;
    int32_t jitter;
    int32_t jitterTaps;
    float jitterRadius;
    int32_t edgeHarden;
    int32_t acneEnabled;
    int32_t acneNormalOffset;
    float acneNormalTexels;
    int32_t acneSlopeScaled;
    float acneSlopeMax;
};

// The four presets.
//
// Resolution is the axis that matters most and costs most -- it is squared, and there are five maps -- so
// it moves first and furthest. The character maps drop faster than the world's because they are redrawn
// every frame while the world's are reused, and because characters are small and close, so their map was
// the one with slack in it.
//
// The acne offset RISES as resolution falls, and that is not a typo: the offset is measured in texels, and
// a smaller map has bigger texels, so a fixed number of texels is a bigger distance in the world. Holding
// it constant across the profiles would leave the low ones striped.
static const ShadowProfile kShadowProfiles[4] = {
    // res  actor  casc  d0 d1 d2  cache  analytic  width  jitter taps radius  harden  acne  nOff  texels  slope  max
    { 1024, 512, 2, 2, 3, 4, 1, 1, 2.0f, 0, 4, 2.0f, 0, 1, 1, 0.9f, 1, 3.5f },  // Baixo
    { 2048, 1024, 3, 1, 2, 3, 1, 1, 2.0f, 0, 6, 2.0f, 0, 1, 1, 0.7f, 1, 3.5f }, // Médio
    { 4096, 2048, 3, 1, 1, 2, 1, 1, 2.0f, 1, 8, 2.0f, 0, 1, 1, 0.6f, 1, 3.5f }, // Alto
    { 4096, 4096, 3, 1, 1, 1, 1, 1, 1.5f, 1, 16, 2.5f, 0, 1, 1, 0.5f, 1, 3.5f } // Ultra
};

// Does the live configuration equal this profile? Floats compared with a tolerance well under the slider
// step, so a value the user never touched still matches after a round trip through the CVar store.
static bool ShadowProfileMatch(const ShadowProfile& p) {
    auto fEq = [](float a, float b) { return std::fabs(a - b) < 0.001f; };
    return CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.Resolution"), 4096) == p.resolution &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.ActorResolution"), 4096) == p.actorResolution &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.CascadeCount"), 3) == p.cascadeCount &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor0"), 1) == p.divisor0 &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor1"), 1) == p.divisor1 &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor2"), 2) == p.divisor2 &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.StaticCache"), 0) == p.staticCache &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdge"), 1) == p.analyticEdge &&
           fEq(CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdgeWidth"), 2.0f), p.analyticEdgeWidth) &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Jitter"), 1) == p.jitter &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterTaps"), 8) == p.jitterTaps &&
           fEq(CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterRadius"), 2.0f), p.jitterRadius) &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHarden"), 0) == p.edgeHarden &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.Enabled"), 1) == p.acneEnabled &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalOffset"), 1) == p.acneNormalOffset &&
           fEq(CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalTexels"), 0.6f), p.acneNormalTexels) &&
           CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeScaled"), 1) == p.acneSlopeScaled &&
           fEq(CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeMax"), 3.5f), p.acneSlopeMax);
}

// Which profile the live configuration IS, or CUSTOM. Read every frame the selector draws, which is what
// makes "Personalizado" appear by itself the moment an advanced slider is moved -- no per-widget callback
// has to remember to set it.
static int32_t ShadowProfileCurrent() {
    for (int32_t i = 0; i < 4; i++) {
        if (ShadowProfileMatch(kShadowProfiles[i])) {
            return i;
        }
    }
    return SHADOW_PROFILE_CUSTOM;
}

static void ShadowProfileApply(int32_t index) {
    if (index < 0 || index >= 4) {
        return; // CUSTOM is a readout, not something to apply
    }
    const ShadowProfile& p = kShadowProfiles[index];
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.Resolution"), p.resolution);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.ActorResolution"), p.actorResolution);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.CascadeCount"), p.cascadeCount);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor0"), p.divisor0);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor1"), p.divisor1);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor2"), p.divisor2);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.StaticCache"), p.staticCache);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdge"), p.analyticEdge);
    CVarSetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdgeWidth"), p.analyticEdgeWidth);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Jitter"), p.jitter);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterTaps"), p.jitterTaps);
    CVarSetFloat(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterRadius"), p.jitterRadius);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHarden"), p.edgeHarden);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.Enabled"), p.acneEnabled);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalOffset"), p.acneNormalOffset);
    CVarSetFloat(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalTexels"), p.acneNormalTexels);
    CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeScaled"), p.acneSlopeScaled);
    CVarSetFloat(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeMax"), p.acneSlopeMax);
    Ship::Context::GetInstance()->GetWindow()->GetGui()->SaveConsoleVariablesNextFrame();
}

// The advanced gate, as free functions so the lambdas below can call them without capturing anything.
static bool ShadowMapOff() {
    return CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode"), SHADOW_MODE_VANILLA) !=
           SHADOW_MODE_SHADOW_MAP;
}
static bool ShadowAdvancedOff() {
    return ShadowMapOff() || !CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Advanced"), 0);
}
static bool ShadowActorAdvancedOff() {
    return CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode"), SHADOW_MODE_VANILLA) != SHADOW_MODE_ACTOR ||
           !CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Advanced"), 0);
}
void SohMenu::AddMenuShadows() {
    AddMenuEntry("Sombras", CVAR_SETTING("Menu.ShadowsSidebarSection"));

    // Shown while any system other than Vanilla is picked -- Vanilla has nothing to tune.
    auto hideUnlessAnyShadowSystem = [](WidgetInfo& info) {
        info.isHidden =
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode"), SHADOW_MODE_VANILLA) == SHADOW_MODE_VANILLA;
    };
    // The stencil silhouettes: their two taste knobs stay out here, the rest is advanced.
    auto hideUnlessShadowsEnabled = [](WidgetInfo& info) {
        info.isHidden =
            CVarGetInteger(CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode"), SHADOW_MODE_VANILLA) != SHADOW_MODE_ACTOR;
    };
    auto actorAdvOnly = [](WidgetInfo& info) { info.isHidden = ShadowActorAdvancedOff(); };
    // Shadow Map, without requiring the advanced switch -- the profile and the darkness slider.
    auto hideUnlessShadowMap = [](WidgetInfo& info) { info.isHidden = ShadowMapOff(); };
    // Everything else. One switch stands between the player and thirty controls.
    auto advOnly = [](WidgetInfo& info) { info.isHidden = ShadowAdvancedOff(); };

    WidgetPath path = { "Sombras", "Geral", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    AddWidget(path, "Shadow System", WIDGET_CVAR_COMBOBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.Mode"))
        .RaceDisable(false)
        .Options(ComboboxOptions()
                     .DefaultIndex(SHADOW_MODE_VANILLA)
                     .ComboMap(shadowModeLabels)
                     .Tooltip("Which shadow system draws. Only one is active at a time.\n\n"
                              "Vanilla: the original game's shadows (Link's feet, the NPC/enemy circles, the "
                              "horse shadow, the sign and snake-statue texture shadows).\n\n"
                              "Actor Shadows: a shape-based drop shadow per actor -- its own silhouette cast "
                              "from the single key light Cel Shading picks, wrapped onto the real ground so it "
                              "follows slopes and bumps. Uses the Cel Shading key selection, but works whether "
                              "or not Cel Shading itself is on.\n\n"
                              "Shadow Map: cascaded depth-map shadows, so the world shadows itself and actors "
                              "cast onto it. Direct3D 11 only -- other backends fall back to Vanilla."));
    AddWidget(path, "Suppress Vanilla Shadows", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.SuppressVanillaShadows"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAnyShadowSystem)
        .Options(CheckboxOptions().DefaultValue(true).Tooltip(
            "Hide the original game's actor shadows (Link's feet, the NPC/enemy circles, the horse shadow, "
            "the sign and snake-statue texture shadows) so only the selected system's shadows show. Turn off "
            "to draw both."));

    AddWidget(path, "Perfil", WIDGET_CVAR_COMBOBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Profile"))
        .RaceDisable(false)
        .PreFunc([](WidgetInfo& info) {
            info.isHidden = ShadowMapOff();
            if (info.isHidden) {
                return;
            }
            // Read back from the settings themselves rather than trusted as stored. A profile is a name for
            // a configuration, not a mode the renderer is in -- so if anything below has been moved by hand
            // (or by the console, or by an older build's saved settings) this says "Personalizado" instead
            // of claiming a preset the game is not actually running.
            const int32_t current = ShadowProfileCurrent();
            if (CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Profile"), SHADOW_PROFILE_CUSTOM) != current) {
                CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Profile"), current);
            }
        })
        .Callback([](WidgetInfo& info) {
            ShadowProfileApply(CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Profile"), SHADOW_PROFILE_HIGH));
        })
        .Options(ComboboxOptions()
                     .DefaultIndex(SHADOW_PROFILE_HIGH)
                     .ComboMap(shadowProfileLabels)
                     .Tooltip("Define de uma vez tudo que decide a QUALIDADE da sombra: tamanho dos mapas, "
                              "quantas faixas, de quantos em quantos quadros cada uma é redesenhada, as "
                              "técnicas de borda e a correção de acne.\n\n"
                              "Baixo: mapas de 1024/512, duas faixas, redesenho espaçado. É o que rende mais "
                              "FPS, e o preço é sombra mais grossa e alcance menor.\n"
                              "Médio: 2048/1024, três faixas.\n"
                              "Alto: 4096/2048 com jitter ligado. O padrão.\n"
                              "Ultra: 4096/4096, tudo redesenhado todo quadro, borda mais fina.\n\n"
                              "\"Personalizado\" não é uma opção que se escolhe: é o que aparece sozinho "
                              "quando algo no Modo Avançado deixa de bater com o perfil. Seus ajustes à mão "
                              "não são desfeitos por isso.\n\n"
                              "NÃO mexe na Intensidade. O quão escura você quer a sombra é gosto, não "
                              "qualidade, e nenhum perfil tem o direito de sobrescrever isso."));

    AddWidget(path, "Intensidade", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.Strength"))
        .RaceDisable(false)
        .PreFunc(hideUnlessShadowMap)
        .Options(FloatSliderOptions()
                     .Tooltip("O quão escura fica uma superfície totalmente na sombra.\n\n0% = nenhuma sombra visível; "
                              "100% = preto.")
                     .Min(0.0f)
                     .Max(1.0f)
                     .DefaultValue(0.5f) // SHADOW_MAP_DEFAULT_STRENGTH
                     .IsPercentage());
    // Snapped to the offered sizes before the widget draws. The combobox looks its current value up with
    // map::at and throws on anything not in the list, and this CVar is reachable from the console and was a
    // free slider in an earlier build -- so a stray value is a crash on opening the menu, not a stray value.
    AddWidget(path, "Opacity", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.Opacity"))
        .RaceDisable(false)
        .PreFunc(hideUnlessShadowsEnabled)
        .Options(FloatSliderOptions()
                     .Tooltip("How dark the shadow's core is. 0 = invisible; higher = darker.")
                     .Min(0.0f)
                     .Max(1.0f)
                     .DefaultValue(0.2f)
                     .IsPercentage());
    AddWidget(path, "Edge Softness", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.EdgeSoftness"))
        .RaceDisable(false)
        .PreFunc(hideUnlessShadowsEnabled)
        .Options(IntSliderOptions()
                     .Tooltip("Smooths the shadow's outline: edge cells the silhouette only partially covers "
                              "render lighter, anti-aliasing the shape. 0 = hard edge; 1 = one lighter step; "
                              "2 = a finer ramp plus a slightly wider fringe.")
                     .Min(0)
                     .Max(2)
                     .DefaultValue(0)
                     .ShowButtons(true)
                     .Format("%d"));

    AddWidget(path, "Modo Avançado", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Advanced"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAnyShadowSystem)
        .Options(CheckboxOptions().DefaultValue(false).Tooltip(
            "Mostra todos os controles individuais, nas seções da barra lateral: o tamanho dos mapas, as "
            "distâncias das faixas, as quatro técnicas de borda, a correção de acne, os divisores de "
            "atualização e as visões de diagnóstico.\n\n"
            "Desligado, aparece só o necessário: qual sistema, qual perfil e o quão escura é a sombra.\n\n"
            "Mexer em qualquer coisa aqui dentro faz o Perfil virar \"Personalizado\" -- o que não desfaz "
            "nada, só para de chamar a configuração por um nome que ela deixou de ter."));
    AddWidget(path, "Restaurar Tudo ao Padrão", WIDGET_BUTTON)
        .PreFunc(hideUnlessAnyShadowSystem)
        .Callback([](WidgetInfo& info) {
            // Clearing each CVar drops it back to the widget's DefaultValue, which is the same value the
            // renderer falls back to -- so this restores the shipped look without hardcoding it twice.
            // Every shadow CVar the three old tabs owned is here; the mode itself is left alone, because
            // "put the settings back" should not also turn the system off.
            static const char* const kAll[] = {
                CVAR_ENHANCEMENT("Graphics.WorldShadows.SuppressVanillaShadows"),
                CVAR_ENHANCEMENT("Graphics.WorldShadows.Opacity"),
                CVAR_ENHANCEMENT("Graphics.WorldShadows.EdgeSoftness"),
                CVAR_ENHANCEMENT("Graphics.WorldShadows.Length"),
                CVAR_ENHANCEMENT("Graphics.WorldShadows.SlabDepth"),
                CVAR_ENHANCEMENT("Graphics.WorldShadows.SlabRise"),
                CVAR_ENHANCEMENT("Graphics.WorldShadows.MaxDistance"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.Strength"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.Resolution"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.ActorResolution"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.CascadeCount"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.Split0"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.Split1"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.Split2"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.MinElevation"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.CasterDrawRadius"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.CasterFirst"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor0"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor1"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor2"),
                CVAR_ENHANCEMENT("Graphics.ShadowMap.BlendFraction"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdge"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdgeWidth"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.Jitter"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterTaps"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterRadius"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHarden"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHardness"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeThreshold"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.StaticCache"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.Layout"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapLevels"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapBase"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapResolution"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderMode"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderLambda"),
                CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderNear"),
                CVAR_ENHANCEMENT("Graphics.ShadowAcne.Enabled"),
                CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalOffset"),
                CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalTexels"),
                CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeScaled"),
                CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeMax"),
                CVAR_DEVELOPER_TOOLS("ShadowMap.ViewSlice"),
            };
            for (const char* cvar : kAll) {
                CVarClear(cvar);
            }
            Ship::Context::GetInstance()->GetWindow()->GetGui()->SaveConsoleVariablesNextFrame();
        })
        .Options(ButtonOptions().Tooltip("Devolve todas as opções de sombra aos valores padrão. Não desliga "
                                         "o sistema nem troca o modo."));

    path = { "Sombras", "Sombras de Ator", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    AddWidget(path, "Length", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.Length"))
        .RaceDisable(false)
        .PreFunc(actorAdvOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("How long the shadow may get. The key light is raised toward straight-overhead "
                              "before projecting, so a low light still casts a short shadow tucked under the "
                              "actor (like the vanilla shadow). Lower = always short and steep; higher = lets "
                              "a low light stretch the shadow out further.")
                     .Format("%.2f")
                     .Min(0.0f)
                     .Max(1.0f)
                     .DefaultValue(0.2f));
    AddWidget(path, "Slab Depth", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.SlabDepth"))
        .RaceDisable(false)
        .PreFunc(actorAdvOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("How far below the feet the shadow conforms to the ground. The shadow is a thin "
                              "stencil 'slab' at the feet that wraps onto whatever ground is inside it. Higher = "
                              "follows ground that dips further (steeper inclines), but past a ledge the shadow "
                              "creeps further down the drop. Lower = clings tight to the feet and won't spill "
                              "over cliff edges, but may clip on steep slopes.")
                     .Format("%.0f")
                     .Min(5.0f)
                     .Max(200.0f)
                     .DefaultValue(8.0f));
    AddWidget(path, "Slab Rise", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.SlabRise"))
        .RaceDisable(false)
        .PreFunc(actorAdvOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("How far ABOVE the feet the shadow can climb onto rising ground. Raise this so the "
                              "shadow still appears where an incline rises higher than the actor's feet (without "
                              "it, the shadow vanishes on up-slopes). Too high starts to catch the actor's own "
                              "lower legs, so keep it just above the ground rise you need.")
                     .Format("%.0f")
                     .Min(0.0f)
                     .Max(120.0f)
                     .DefaultValue(8.0f));
    AddWidget(path, "Render Distance: %d", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.WorldShadows.MaxDistance"))
        .RaceDisable(false)
        .PreFunc(actorAdvOnly)
        .Options(IntSliderOptions()
                     .Tooltip("Performance: actors farther than this from the camera get no shape shadow (each "
                              "shadow rebuilds and redraws the actor's whole silhouette, so distant ones cost "
                              "more than they're worth). Lower to gain frames in crowded scenes; raise for "
                              "shadows that stay visible into the distance.")
                     .Min(300)
                     .Max(5000)
                     .DefaultValue(550)
                     .ShowButtons(true)
                     .Format("%d"));
    AddWidget(path, "Debug", WIDGET_SEPARATOR_TEXT).PreFunc(actorAdvOnly);
    AddWidget(path, "Show Shadow Volume", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_DEVELOPER_TOOLS("WorldShadows.ShowVolume"))
        .PreFunc(actorAdvOnly)
        .Options(CheckboxOptions().Tooltip(
            "Draws the actual 3D shadow volume translucently so you can see its shape: black top/bottom caps, "
            "blue side walls. The ground inside this volume is what gets shadowed."));

    path = { "Sombras", "Mapa e Faixas", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    AddWidget(path, "Resolução", WIDGET_CVAR_COMBOBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.Resolution"))
        .RaceDisable(false)
        .PreFunc([](WidgetInfo& info) {
            info.isHidden = ShadowAdvancedOff();
            if (info.isHidden) {
                return;
            }
            const int32_t offered[] = { 512, 1024, 2048, 4096 };
            int32_t current = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.Resolution"), 4096);
            int32_t nearest = offered[0];
            for (int32_t candidate : offered) {
                if (std::abs(candidate - current) < std::abs(nearest - current)) {
                    nearest = candidate;
                }
            }
            if (nearest != current) {
                CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.Resolution"), nearest);
            }
        })
        .Options(ComboboxOptions()
                     .DefaultIndex(4096) // SHADOW_MAP_DEFAULT_RESOLUTION
                     .ComboMap(shadowMapResolutionLabels)
                     .Tooltip("Tamanho do mapa de cada faixa, por lado. É o maior controle de qualidade "
                              "que existe aqui: toda borda de sombra é desenhada nessa grade, então dobrar "
                              "esse número reduz à metade o tamanho dos degraus no contorno da sombra.\n\n"
                              "Também é o maior custo, e ele cresce com o QUADRADO do número. Cada faixa "
                              "ganha o próprio mapa nesse tamanho, e existem dois conjuntos, um para o "
                              "cenário e outro para os personagens: cinco mapas no total. Em 4096 isso dá "
                              "cerca de 168 MB de memória de vídeo, contra 10 MB em 1024. Se o jogo estiver "
                              "pesado, baixe esta opção primeiro."));
    // Same snapping guard as the world layer's, and for the same reason: the combobox throws on a value
    // that is not in its map, and this CVar is reachable from the console.
    AddWidget(path, "Resolução (Personagens)", WIDGET_CVAR_COMBOBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.ActorResolution"))
        .RaceDisable(false)
        .PreFunc([](WidgetInfo& info) {
            info.isHidden = ShadowAdvancedOff();
            if (info.isHidden) {
                return;
            }
            const int32_t offered[] = { 512, 1024, 2048, 4096 };
            int32_t current = CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.ActorResolution"),
                                             SHADOW_MAP_DEFAULT_ACTOR_RESOLUTION);
            int32_t nearest = offered[0];
            for (int32_t candidate : offered) {
                if (std::abs(candidate - current) < std::abs(nearest - current)) {
                    nearest = candidate;
                }
            }
            if (nearest != current) {
                CVarSetInteger(CVAR_ENHANCEMENT("Graphics.ShadowMap.ActorResolution"), nearest);
            }
        })
        .Options(ComboboxOptions()
                     .DefaultIndex(SHADOW_MAP_DEFAULT_ACTOR_RESOLUTION)
                     .ComboMap(shadowMapResolutionLabels)
                     .Tooltip("Tamanho do mapa das sombras dos PERSONAGENS, escolhido à parte do cenário.\n\n"
                              "Vale baixar esta antes da outra. Os mapas dos personagens são redesenhados "
                              "todo quadro, porque os personagens se mexem, enquanto os do cenário são "
                              "reaproveitados enquanto a câmera fica parada — e o custo de um mapa é o "
                              "mesmo quer tenha muita coisa dentro ou pouca, porque limpá-lo já escreve a "
                              "superfície inteira.\n\n"
                              "O que se perde é pouco: personagens são pequenos, ficam perto e projetam no "
                              "chão logo à frente, então o mapa deles já estava sobrando em 4096. Nunca "
                              "passa da resolução do cenário; igualar as duas desliga a separação."));
    AddWidget(path, "Quantidade de Faixas: %d", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.CascadeCount"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(IntSliderOptions()
                     .Tooltip("Quantas das três faixas de distância abaixo são realmente construídas.\n\n"
                              "Isto NÃO é um ajuste de qualidade. As distâncias são absolutas, então "
                              "diminuir a quantidade não espalha o mesmo alcance por menos mapas: ele CORTA o "
                              "alcance na última faixa ativa, e as sombras simplesmente somem dali para "
                              "frente. Diminua para ganhar FPS, sabendo que as sombras distantes vão junto.")
                     .Min(1)
                     .Max(3)
                     .DefaultValue(3) // SHADOW_MAP_DEFAULT_CASCADES
                     .ShowButtons(true)
                     .Format("%d"));

    AddWidget(path, "Distâncias", WIDGET_SEPARATOR_TEXT).PreFunc(advOnly);
    AddWidget(path, "Faixa Próxima Termina Em: %.0f", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.Split0"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("Onde a primeira faixa termina, em unidades do mundo (o Link tem cerca de "
                              "60 de altura).\n\n"
                              "Todas as faixas recebem um mapa da mesma resolução, então uma faixa mais "
                              "CURTA gasta esse mapa em menos chão e as sombras dela saem mais nítidas. Esta "
                              "primeira cobre o que está em volta do Link, então é ela que decide o quanto a "
                              "sombra dele fica definida. Diminua para sombras mais nítidas de perto.")
                     .Format("%.0f")
                     .Min(50.0f)
                     .Max(600.0f)
                     .DefaultValue(350.0f)); // SHADOW_MAP_DEFAULT_SPLIT_0
    AddWidget(path, "Faixa Média Termina Em: %.0f", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.Split1"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("Onde a segunda faixa termina. Mantenha as distâncias em ordem crescente: se "
                              "uma ficar abaixo da faixa anterior, o renderizador a empurra de volta para cima "
                              "e aquela faixa acaba não cobrindo nada.\n\n"
                              "Este é o controle de nitidez da distância média. O mapa de uma faixa é "
                              "esticado sobre a largura da visão na borda MAIS DISTANTE dela, então a última "
                              "faixa fica grosseira independentemente de onde comece. Empurrar esta para mais "
                              "longe entrega mais da cena para uma faixa que ainda está boa, ao custo de a "
                              "última começar mais tarde.")
                     .Format("%.0f")
                     .Min(100.0f)
                     .Max(3000.0f)
                     .DefaultValue(2500.0f)); // SHADOW_MAP_DEFAULT_SPLIT_1
    AddWidget(path, "Faixa Distante Termina Em: %.0f", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.Split2"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("Onde a terceira faixa termina. Normalmente é nela que cai a sombra de um "
                              "prédio, então vale apertar se as sombras de média distância estiverem "
                              "grosseiras.\n\n"
                              "É também onde TODA sombra acaba: a última faixa ativa é a distância de "
                              "desenho, e com as três faixas padrão é esta. Aumentar estica o mesmo mapa "
                              "sobre mais chão, então as sombras distantes ficam mais quadriculadas em vez "
                              "de melhores. Diminuir deixa tudo dentro do novo alcance mais nítido e "
                              "simplesmente termina as sombras mais cedo.")
                     .Format("%.0f")
                     .Min(300.0f)
                     .Max(12000.0f)
                     .DefaultValue(6000.0f)); // SHADOW_MAP_DEFAULT_SPLIT_2

    AddWidget(path, "Luz e Alcance", WIDGET_SEPARATOR_TEXT).PreFunc(advOnly);
    AddWidget(path, "Altura Mínima do Sol", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.MinElevation"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("Quão baixo o sol pode ficar antes de as sombras serem projetadas a partir "
                              "dele, medido como o seno do ângulo acima do horizonte (0,5 é trinta graus).\n\n"
                              "Um sol na linha do horizonte estica toda sombra até o infinito, o que parece "
                              "errado bem antes de ser geometricamente errado, e ainda desperdiça os mapas "
                              "numa área muito maior que a cena. Só a altura é levantada: a direção para "
                              "onde as sombras apontam continua a mesma.")
                     .Format("%.2f")
                     .Min(0.1f)
                     .Max(0.95f)
                     .DefaultValue(0.60f)); // SHADOW_MAP_DEFAULT_MIN_ELEVATION
    AddWidget(path, "Alcance Fora da Tela: %.0f", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.CasterDrawRadius"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("Até que distância fora da tela um objeto continua sendo desenhado apenas "
                              "para poder projetar sombra.\n\n"
                              "Normalmente o jogo para de desenhar o que a câmera não vê, mas uma árvore "
                              "atrás de você ainda joga a sombra dela na sua frente. Sem isso, as sombras "
                              "piscariam a cada vez que você virasse a câmera. O teste segue a luz em vez de "
                              "usar um círculo simples, então esta é a margem em volta do caminho da sombra, "
                              "não em volta do objeto.\n\n"
                              "Diminua para ganhar FPS em áreas abertas; o preço é pontos de referência "
                              "distantes perderem a sombra ao sair da tela.")
                     .Format("%.0f")
                     .Min(400.0f)
                     .Max(4000.0f)
                     .DefaultValue(1500.0f)); // SHADOW_MAP_DEFAULT_CASTER_DRAW_RADIUS

    AddWidget(path, "Desenhar Atores Antes da Sala", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.CasterFirst"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(CheckboxOptions().DefaultValue(false).Tooltip(
            "Inverte a ordem do quadro: o laço de atores desenha ANTES da sala.\n\n"
            "Antes isto arrastava junto o fluxo TRANSLÚCIDO dos atores, e translúcido compõe por ordem de "
            "envio, não por profundidade -- era assim que a Navi aparecia através do piso no Cemitério. "
            "Agora só o fluxo opaco troca de lugar: o translúcido dos atores é desviado por saltos e "
            "executa depois da sala, exatamente como no original.\n\n"
            "Continua DESLIGADO por padrão até a correção ser confirmada em jogo.\n\n"
            "Serve para tirar um quadro de atraso da sombra de quem está em movimento. Sem isso, a sombra "
            "de um personagem correndo fica sempre onde ele estava um quadro atrás -- invisível no cenário, "
            "que não anda, e visível nele.\n\n"
            "O custo é que atores e sala passam a ser enviados em ordem trocada, e ordem de envio é o que "
            "decide a sobreposição de geometria translúcida. É a ÚNICA coisa que o modo Shadow Map muda na "
            "ordem de desenho.\n\n"
            "O que se ganha é frescor na sombra de quem ESTÁ SE MOVENDO -- invisível no cenário, porque "
            "cenário não anda.\n\n"
            "Se algo ainda aparecer empilhado errado só com Shadow Map ligado, isto continua sendo a "
            "primeira coisa a desligar."));

    path = { "Sombras", "Desempenho", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    AddWidget(path, "Desempenho", WIDGET_SEPARATOR_TEXT).PreFunc(advOnly);
    // Shared tooltip tail: the trade-off is identical for all three, only the cascade differs. A macro
    // rather than a variable because Tooltip() keeps the raw pointer it is handed -- a std::string built
    // per widget would be freed before the menu ever draws it, while adjacent literals are joined by the
    // compiler and live in static storage.

    // SOH [Enhancement] The three sliders below are CASCADE-ONLY, and hidden rather than left inert in the
    // clipmap layout.
    //
    // The divisor is read in the cascade fit, inside the `else` of the layout branch (see RenderShadowMap in
    // interpreter.cpp): it freezes a band's matrix on a frame it is not due, and the content key then agrees
    // the slice already holds what a redraw would produce. The clipmap has no equivalent and needs none --
    // it snaps each level's centre to that level's own texel, so a camera that has not crossed a texel
    // produces a bit-identical matrix and the slice is reused with no parking code at all.
    //
    // So in clipmap these read as sliders that do nothing, which is how they were reported. Reuse still
    // happens there; it is simply not what this number controls.
    auto hideUnlessCascadeLayout = [](WidgetInfo& info) {
        info.isHidden =
            ShadowAdvancedOff() || CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Layout"), 0) != 0;
    };
#define SHADOW_UPDATE_RATE_TOOLTIP_TAIL                                                                        \
    "\n\nPular a reconstrução CONGELA a faixa inteira, matriz inclusive. O mapa guardado foi desenhado com " \
    "a matriz daquele quadro, e lê-lo com outra projetaria a sombra a partir de onde a luz estava — uma "   \
    "sombra atrasada vira uma sombra no lugar errado, que é pior do que o custo economizado. Por isso o "     \
    "preço aqui é atraso, não deslocamento: a sombra desta faixa reage um quadro depois, e isso aparece "   \
    "principalmente ao girar a câmera rápido."
    AddWidget(path, "Atualização da Faixa Próxima: 1 a cada %d quadros", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor0"))
        .RaceDisable(false)
        .PreFunc(hideUnlessCascadeLayout)
        .Options(IntSliderOptions()
                     .Tooltip("Com que frequência a faixa mais próxima é redesenhada. 1 é todo quadro "
                              "(60 Hz a 60 fps), 2 é um sim um não (30 Hz).\n\n"
                              "Deixe em 1. Esta é a faixa que segue você de perto e é onde o olho está; é "
                              "também a mais barata, porque cobre pouco chão e pega poucos objetos. "
                              "Economizar aqui rende quase nada e o atraso é visto "
                              "imediatamente." SHADOW_UPDATE_RATE_TOOLTIP_TAIL)
                     .Min(1)
                     .Max(4)          // SHADOW_MAP_MAX_CASCADE_DIVISOR
                     .DefaultValue(1) // SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_0
                     .ShowButtons(true)
                     .Format("%d"));
    AddWidget(path, "Atualização da Faixa Média: 1 a cada %d quadros", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor1"))
        .RaceDisable(false)
        .PreFunc(hideUnlessCascadeLayout)
        .Options(IntSliderOptions()
                     .Tooltip("Com que frequência a faixa média é redesenhada. 1 é todo quadro (60 Hz a 60 "
                              "fps), 2 é um sim um não (30 Hz).\n\n"
                              "É o meio-termo dos dois lados: cobre bastante cena, mas ainda perto o "
                              "suficiente para o atraso ser notado em objetos que se movem. Suba para 2 só "
                              "depois de já ter subido a faixa distante e ainda precisar de "
                              "FPS." SHADOW_UPDATE_RATE_TOOLTIP_TAIL)
                     .Min(1)
                     .Max(4)          // SHADOW_MAP_MAX_CASCADE_DIVISOR
                     .DefaultValue(1) // SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_1
                     .ShowButtons(true)
                     .Format("%d"));
    AddWidget(path, "Atualização da Faixa Distante: 1 a cada %d quadros", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.UpdateDivisor2"))
        .RaceDisable(false)
        .PreFunc(hideUnlessCascadeLayout)
        .Options(IntSliderOptions()
                     .Tooltip("Com que frequência a faixa distante é redesenhada. O padrão é 2 — 30 Hz a "
                              "60 fps, enquanto as outras duas ficam em 60 Hz.\n\n"
                              "É a faixa que vale reduzir, e por dois motivos ao mesmo tempo: ela cobre a "
                              "maior área com o mesmo número de células, então o conteúdo dela é o que "
                              "menos muda de um quadro para o outro (uma árvore a três mil unidades anda "
                              "uma fração de célula), e é a mais cara de preencher, porque a área grande "
                              "varre o maior número de objetos." SHADOW_UPDATE_RATE_TOOLTIP_TAIL)
                     .Min(1)
                     .Max(4)          // SHADOW_MAP_MAX_CASCADE_DIVISOR
                     .DefaultValue(2) // SHADOW_MAP_DEFAULT_CASCADE_DIVISOR_2
                     .ShowButtons(true)
                     .Format("%d"));
#undef SHADOW_UPDATE_RATE_TOOLTIP_TAIL

    path = { "Sombras", "Depuração", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    AddWidget(path, "Depuração", WIDGET_SEPARATOR_TEXT).PreFunc(advOnly);
    // Live readout of which light won the frame. A lot of policy decides the single direction the cascades
    // get, so without this a hierarchy that picked the wrong light is indistinguishable from one that
    // picked the right light and aimed it badly. Same live-name trick as the caster census below.
    // The CVar key still says ShowCascadeBounds because that is what the first view did and renaming it
    // would silently reset everyone's saved value. The label does not, because the slider selects between
    // several views and only one of them is about cascade bounds.
    AddWidget(path, "Visão de Diagnóstico: %d", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_DEVELOPER_TOOLS("ShadowMap.ShowCascadeBounds"))
        .PreFunc(advOnly)
        .Options(IntSliderOptions()
                     .Tooltip("0 = desligado.\n\n"
                              "As visões 1 e 2 mostram o que o sistema de sombras PRODUZIU. As visões 3, 4, "
                              "5 e 7 mostram o que ele RECEBEU -- use estas quando a sombra sai com a FORMA "
                              "errada (facetada, triangular, escadinha) em vez de no lugar errado. Um "
                              "ajuste age sobre o RESULTADO da comparação, então consegue deixar esse tipo "
                              "de defeito menos visível e nunca consegue dizer de onde ele veio.\n\n"
                              "1 = pinta tudo que está FORA da área de uma faixa como totalmente sombreado. "
                              "Uma superfície fora dela é silenciosamente considerada iluminada, então uma "
                              "sombra que para no limite da faixa fica idêntica a uma que nunca foi "
                              "projetada. Esta é a única forma de distinguir as duas.\n\n"
                              "2 = colore as duas camadas de projeção em vez de sombrear com elas. VERDE "
                              "onde o cenário bloqueia a luz, VERMELHO onde um personagem bloqueia. Use para "
                              "descobrir se algo está sendo capturado e em qual camada.\n\n"
                              "3 = a normal da superfície, como cor. Manchas chapadas de uma cor só, numa "
                              "superfície que deveria variar suavemente, são os próprios triângulos da "
                              "malha aparecendo.\n\n"
                              "4 = de onde veio essa normal. VERDE = a normal de vértice do desenho, "
                              "escurecendo conforme a normal interpolada encurta. VERMELHO = o desenho não "
                              "tem normal, então é usada uma normal de face recuperada das derivadas de "
                              "tela -- constante ao longo de um triângulo inteiro.\n\n"
                              "5 = a cobertura crua do filtro, antes de a definição de borda reescrevê-la. "
                              "Compare com a imagem sombreada: facetado aqui também significa que a "
                              "comparação ou o mapa desenhou o defeito; liso aqui significa que a definição "
                              "de borda desenhou.\n\n"
                              "7 = em qual faixa cada pixel caiu: vermelho, verde e azul, da mais próxima "
                              "para a mais distante. Use para incluir ou descartar a escolha de faixa.\n\n"
                              "(6, 8 e 9 não existem mais: mediam mecanismos que foram removidos. A "
                              "numeração das outras foi mantida de propósito, para 5 continuar significando "
                              "o que significava.)\n\n"
                              "A névoa é desligada em todas as visões, para a distância não lavar as "
                              "cores.\n\n"
                              "Qualquer valor diferente de zero também preenche a lista abaixo.")
                     .Min(0)
                     .Max(7) // SHADOW_MAP_MAX_DEBUG_VIEW, written out per the note at the top of this panel
                     .DefaultValue(0)
                     .ShowButtons(true)
                     .Format("%d"));
    // SOH [Enhancement] Timing without the debug picture. The slider above always timed the pass as a side
    // effect, but every one of its values also repaints the scene, so the only frames that could be measured
    // were frames that no longer looked like the game -- and the numbers were about those frames. This is the
    // same measurement with nothing drawn differently.
    AddWidget(path, "Medir Custo na GPU (log)", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_DEVELOPER_TOOLS("ShadowMap.ProfileGpu"))
        .PreFunc(advOnly)
        .Options(CheckboxOptions().Tooltip(
            "Mede quanto tempo de GPU o mapa de sombras consome e escreve o resultado no log a cada segundo. "
            "Não altera nada do que é desenhado, então os números valem para o jogo como ele realmente é.\n\n"
            "A linha traz o custo do quadro inteiro, o custo do passe de profundidade dentro dele, em quantos "
            "quadros o passe precisou ser submetido e QUANTAS FATIAS foram redesenhadas por quadro. Essa última "
            "é a que decide o que fazer: com poucas fatias por quadro o passe já está sendo reaproveitado e o "
            "custo restante é geometria; com quase todas, algo está impedindo o reaproveitamento.\n\n"
            "Somente Direct3D 11."));
    AddWidget(path, "Ver o Mapa de Profundidade: %d", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_DEVELOPER_TOOLS("ShadowMap.ViewSlice"))
        .PreFunc(advOnly)
        .Options(IntSliderOptions()
                     .Tooltip("Desenha uma fatia do mapa de profundidade num canto da tela. 0 = desligado.\n\n"
                              "Todas as Visões de Diagnóstico acima olham o RECEPTOR — o que o pixel sendo "
                              "sombreado recebeu. Esta mostra o que o passo de profundidade GUARDOU, que é a "
                              "outra metade do sistema. Um defeito de sombra pode morar em qualquer uma das "
                              "duas, e até agora só uma podia ser inspecionada.\n\n"
                              "1 em diante são as faixas da camada do MUNDO, em ordem; depois delas vêm as da "
                              "camada de ATORES, que é mais curta.\n\n"
                              "AZUL é célula vazia — nada foi desenhado ali. Isso é diferente de cinza "
                              "escuro, que é algo próximo: os dois são o mesmo número no mapa e significam "
                              "coisas opostas, e confundir um com o outro faz um caster ausente parecer "
                              "presente.\n\n"
                              "A amostragem é por ponto de propósito. Filtrar borraria vizinhos numa imagem "
                              "que a comparação de sombra nunca enxerga.")
                     .Min(0)
                     .Max(5)
                     .DefaultValue(0)
                     .ShowButtons(true)
                     .Format("%d"));
    AddWidget(path, "Salvar captura das sombras (DirectX)", WIDGET_BUTTON)
        .PreFunc(advOnly)
        .Options(
            ButtonOptions().Tooltip("Salva as profundidades e configurações da camada do cenário em shadow-captures, "
                                    "na pasta de dados do jogo. Pode causar uma pausa durante a leitura da GPU. "
                                    "Clique com o defeito visível; não altera a aparência das sombras."))
        .Callback([](WidgetInfo& info) {
            CVarSetString(SHADOW_MAP_CAPTURE_STATUS_CVAR, "Aguardando um quadro com Shadow Map no DirectX...");
            // What produced the file, written BEFORE the request and from here rather than from a render
            // hook.
            //
            // The renderer copies this CVar into the capture's game_context, and it has to be there by the
            // time the capture is written. This button runs from inside the graphics command run, because
            // that is where the menu is drawn, so a click on a frame whose shadow pass has already gone by
            // has the backend writing the file that same frame. Anything that fills the context later is
            // too late, and the default "{}" is what lands on disk.
            //
            // The build identity first: which binary a capture came from is the question it most often has
            // to answer, and the one no amount of staring at the depths can recover.
            nlohmann::json context;
            context["git_commit"] = (const char*)gGitCommitHash;
            context["git_branch"] = (const char*)gGitBranch;
            context["build_version"] = (const char*)gBuildVersion;
            context["scene_id"] = gPlayState != nullptr ? int(gPlayState->sceneNum) : -1;
            context["day_time_u16"] = gPlayState != nullptr ? int(gSaveContext.dayTime) : -1;
            context["minimum_elevation"] =
                CVarGetFloat(CVAR_ENHANCEMENT("Graphics.ShadowMap.MinElevation"),
                             0.60f); // SHADOW_MAP_DEFAULT_MIN_ELEVATION, spelled as the sliders above do
            CVarSetString(SHADOW_MAP_CAPTURE_CONTEXT_CVAR, context.dump().c_str());
            CVarSetInteger(SHADOW_MAP_CAPTURE_REQUEST_CVAR, 1);
        });
    AddWidget(path, "Captura das sombras", WIDGET_TEXT).PreFunc([](WidgetInfo& info) {
        info.isHidden = ShadowAdvancedOff();
        info.name = CVarGetString(SHADOW_MAP_CAPTURE_STATUS_CVAR, "");
    });
    // Live list of what the world (green) caster layer is actually made of, so a stray green blob in the
    // debug view can be named instead of guessed at. WIDGET_TEXT draws widget.name, and PreFunc runs first,
    // so rewriting the name each frame is what makes it live.
    AddWidget(path, "Cenário projetando (média/frame):", WIDGET_TEXT).RaceDisable(false).PreFunc([](WidgetInfo& info) {
        info.isHidden =
            ShadowAdvancedOff() || CVarGetInteger(CVAR_DEVELOPER_TOOLS("ShadowMap.ShowCascadeBounds"), 0) == 0;
        if (!info.isHidden) {
            const char* census = ToonLighting_ShadowMapCasterCensus();
            info.name = std::string("Cenário projetando (média/frame):\n") + (census != nullptr ? census : "");
        }
    });

    path = { "Sombras", "Borda", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    auto hideUnlessAnalytic = [](WidgetInfo& info) {
        info.isHidden =
            ShadowAdvancedOff() || !CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdge"), 0);
    };

    AddWidget(path, "Ativar Borda Analítica", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdge"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(CheckboxOptions().DefaultValue(true).Tooltip(
            "Calcula onde a borda da sombra CRUZA o texel, em vez de misturar as quatro comparações.\n\n"
            "Usa exatamente as mesmas quatro leituras que já são feitas: não custa nenhuma leitura extra "
            "de textura, só aritmética.\n\n"
            "É a opção mais barata da aba e a que ataca o serrilhado diretamente. Exata para uma borda reta "
            "(paredes, degraus, telhados, plataformas); em folhagem ela volta sozinha ao comportamento "
            "antigo."));
    AddWidget(path, "Largura da Rampa: %.2f texels", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.AnalyticEdgeWidth"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAnalytic)
        .Options(FloatSliderOptions()
                     .Tooltip("Quantos texels a transição ocupa.\n\n"
                              "1.00 é a resposta geométrica: a rampa ocupa exatamente o texel que a borda "
                              "atravessa. Acima disso a borda é suavizada de propósito -- é o amaciamento "
                              "mais barato do sistema, porque custa conta e nenhuma leitura.")
                     .Min(0.25f)
                     .Max(4.0f) // SHADOW_MAP_MAX_ANALYTIC_EDGE_WIDTH
                     .Step(0.05f)
                     .DefaultValue(2.0f) // SHADOW_MAP_DEFAULT_ANALYTIC_EDGE_WIDTH
                     .Format("%.2f"));

    // ===========================================================================================
    // Technique 3 -- stochastic jitter.
    // ===========================================================================================
    path = { "Sombras", "Borda", SECTION_COLUMN_1 };

    auto hideUnlessJitter = [](WidgetInfo& info) {
        info.isHidden = ShadowAdvancedOff() || !CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Jitter"), 0);
    };

    AddWidget(path, "Ativar Jitter", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Jitter"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(CheckboxOptions().DefaultValue(true).Tooltip(
            "Espalha as amostras num disco girado por um ângulo diferente em cada pixel.\n\n"
            "O degrau não fica menor, mas pixels vizinhos param de pular no mesmo lugar, então a borda é "
            "lida como granulado em vez de escada.\n\n"
            "ATENÇÃO: este jogo tem FXAA e não tem filtro temporal. Sem nada para calcular a média do "
            "granulado, ele aparece como chiado -- especialmente em movimento. Aumente as Amostras se "
            "incomodar."));
    AddWidget(path, "Amostras: %d", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterTaps"))
        .RaceDisable(false)
        .PreFunc(hideUnlessJitter)
        .Options(IntSliderOptions()
                     .Tooltip("Quantas leituras o disco faz.\n\n"
                              "É o único ajuste desta aba que custa banda de memória de verdade, e é também "
                              "o que decide se o resultado é lido como suavidade ou como chiado. Mais "
                              "amostras = mais suave e mais caro.")
                     .Min(1)
                     .Max(16)           // SHADOW_MAP_MAX_JITTER_TAPS
                     .DefaultValue(8)); // SHADOW_MAP_DEFAULT_JITTER_TAPS
    AddWidget(path, "Raio: %.2f texels", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.JitterRadius"))
        .RaceDisable(false)
        .PreFunc(hideUnlessJitter)
        .Options(FloatSliderOptions()
                     .Tooltip("Tamanho do disco, em texels da cascata amostrada.\n\n"
                              "É o que define a largura da penumbra neste modo. Raio 0 desliga o efeito: "
                              "todas as amostras cairiam no mesmo lugar.")
                     .Min(0.0f)
                     .Max(8.0f) // SHADOW_MAP_MAX_JITTER_RADIUS
                     .Step(0.1f)
                     .DefaultValue(2.0f) // SHADOW_MAP_DEFAULT_JITTER_RADIUS
                     .Format("%.2f"));

    // ===========================================================================================
    // Edge hardening -- the control in the other direction from everything above.
    // ===========================================================================================
    path = { "Sombras", "Borda", SECTION_COLUMN_1 };

    auto hideUnlessHarden = [](WidgetInfo& info) {
        info.isHidden =
            ShadowAdvancedOff() || !CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHarden"), 0);
    };

    AddWidget(path, "As outras seções alargam a borda. Esta comprime.", WIDGET_TEXT).PreFunc(advOnly);
    AddWidget(path, "A borda se identifica sozinha: cobertura vale 0 ou 1 em todo o interior, e só", WIDGET_TEXT)
        .PreFunc(advOnly);
    AddWidget(path, "o contorno cai no meio. Então isto age só no contorno, sem procurar por ele", WIDGET_TEXT)
        .PreFunc(advOnly);
    AddWidget(path, "e sem nenhuma leitura extra de textura.", WIDGET_TEXT).PreFunc(advOnly);

    AddWidget(path, "Ativar Dureza da Borda", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHarden"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(CheckboxOptions().DefaultValue(false).Tooltip(
            "Comprime a faixa de transição da sombra para um contorno mais definido.\n\n"
            "Age depois de tudo o mais, sobre o valor final de cobertura -- então funciona igual com "
            "qualquer combinação das outras técnicas, e a visão de diagnóstico 5 continua mostrando o "
            "valor CRU, antes desta compressão."));
    AddWidget(path, "Dureza", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeHardness"))
        .RaceDisable(false)
        .PreFunc(hideUnlessHarden)
        .Options(FloatSliderOptions()
                     .Tooltip("0% deixa a cobertura exatamente como chegou. 100% é um corte seco, sem "
                              "transição nenhuma.\n\n"
                              "Corte seco nem sempre é o que se quer: a penumbra carrega o tamanho do texel "
                              "da faixa, então tirá-la tira a única pista de que a distância está sendo "
                              "amostrada mais grosso -- e o contorno passa a mostrar a grade de texels que "
                              "ela escondia. O ponto costuma ficar um pouco antes de 100%.")
                     .Min(0.0f)
                     .Max(1.0f)
                     .Step(0.01f)
                     .DefaultValue(0.5f) // SHADOW_MAP_DEFAULT_EDGE_HARDNESS
                     .IsPercentage());
    AddWidget(path, "Limiar", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.EdgeThreshold"))
        .RaceDisable(false)
        .PreFunc(hideUnlessHarden)
        .Options(FloatSliderOptions()
                     .Tooltip("Onde fica o contorno dentro da faixa de cobertura. 50% é a resposta "
                              "geométrica: metade do kernel bloqueada é a borda.\n\n"
                              "Mover ENGROSSA ou AFINA a sombra. Abaixo de 50% um pixel pouco bloqueado já "
                              "conta como sombra e ela se espalha; acima, ela se recolhe. É com isto, junto "
                              "com a Dureza, que se engrossa um contorno em vez de só afiá-lo.")
                     .Min(0.05f)
                     .Max(0.95f)
                     .Step(0.01f)
                     .DefaultValue(0.5f) // SHADOW_MAP_DEFAULT_EDGE_THRESHOLD
                     .IsPercentage());

    // ===========================================================================================
    // Layout -- the ladder, or the clipmap. The one setting here that changes the SHAPE of the system.
    // ===========================================================================================

    path = { "Sombras", "Forma do Mapa", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    auto hideUnlessClipmap = [](WidgetInfo& info) {
        info.isHidden = ShadowAdvancedOff() || CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Layout"), 0) !=
                                                   1; // SHADOW_MAP_LAYOUT_CLIPMAP
    };

    AddWidget(path, "Formato", WIDGET_CVAR_COMBOBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.Layout"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(ComboboxOptions()
                     .ComboMap(shadowLayoutLabels)
                     .DefaultIndex(0) // SHADOW_MAP_DEFAULT_LAYOUT
                     .Tooltip("Como o mapa é distribuído pelo mundo. É a única opção desta aba que muda a FORMA "
                              "do sistema, não o acabamento dele.\n\n"
                              "CASCATAS: faixas ajustadas a fatias do campo de visão. O que existe hoje. A razão "
                              "de texel entre faixas vizinhas é de 3 a 8 vezes, e é esse salto que a transição "
                              "precisa esconder.\n\n"
                              "CLIPMAP: quadrados aninhados centrados na CÂMERA, cada um com o dobro da extensão "
                              "do anterior. A razão de texel entre níveis vizinhos é exatamente 2, e as "
                              "fronteiras são círculos que andam COM você em vez de varrerem o mundo quando a "
                              "câmera gira.\n\n"
                              "É a metade direcional do Virtual Shadow Map da Unreal -- a parte que dá para "
                              "portar. A paginação de memória dela não dá: precisa de compute, indirect draw e "
                              "um passe de profundidade prévio.\n\n"
                              "TROCA: o nível 0 do clipmap é mais grosso que a cascata 0 (0,37 contra 0,09 "
                              "unidade por texel a 1024). A cascata 0 é fina além do que se enxerga àquela "
                              "distância; é justamente esse desperdício que o clipmap remove. Se a sombra aos "
                              "pés do Link ficar grosseira, suba a Resolução."));
    AddWidget(path, "Níveis: %d", WIDGET_CVAR_SLIDER_INT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapLevels"))
        .RaceDisable(false)
        .PreFunc(hideUnlessClipmap)
        .Options(IntSliderOptions()
                     .Tooltip("Quantos quadrados aninhados. Cada um dobra a extensão do anterior, então N "
                              "níveis alcançam a extensão base vezes 2^(N-1).\n\n"
                              "É AQUI que se ganha alcance. Níveis custam LINEAR e compram alcance "
                              "EXPONENCIAL: um nível a mais dobra o alcance pelo preço de uma fatia -- e uma "
                              "fatia de clipmap é pequena, porque densidade uniforme significa que nenhum "
                              "nível precisa ser grande para ser nítido.\n\n"
                              "Aumentar a Extensão do Nível 0 também estende o alcance, mas às custas da "
                              "nitidez perto. Mais níveis não tem esse custo.")
                     .Min(1)
                     .Max(10)           // SHADOW_MAP_MAX_CLIPMAP_LEVELS
                     .DefaultValue(8)); // SHADOW_MAP_DEFAULT_CLIPMAP_LEVELS
    AddWidget(path, "Cache de Casters Estáticos", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.StaticCache"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(CheckboxOptions().DefaultValue(false).Tooltip(
            "Guarda uma segunda cópia de cada fatia do mundo contendo SÓ o que não se mexe, e a cada quadro "
            "copia essa cópia e redesenha apenas o que se moveu por cima.\n\n"
            "O caso que isto resolve: hoje uma árvore balançando custa a malha da sala inteira de novo, em "
            "toda fatia que a árvore alcança, todo quadro em que ela se mexe.\n\n"
            "A cópia substitui tanto a limpeza quanto a rasterização da sala. A 4096 uma fatia custa ~420 µs "
            "para redesenhar e a cópia custa ~62 µs; a 1024 a cópia custa ~4 µs, que é nada.\n\n"
            "CUSTA MEMÓRIA -- um segundo array do tamanho da camada do mundo:\n"
            "  cascatas a 4096: +96 MB     cascatas a 2048: +24 MB\n"
            "  clipmap a 1024:  +12 MB     clipmap a 2048:  +48 MB\n\n"
            "Por isso combina com o Clipmap: o formato que quer muitos níveis pequenos é justamente aquele "
            "onde a segunda cópia sai barata.\n\n"
            "NÃO ajuda cena parada -- essa já não redesenha nada. Isto compra de volta o custo do "
            "movimento, não o de ficar parado."));

    AddWidget(path, "Extensão do Nível 0: %.0f", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapBase"))
        .RaceDisable(false)
        .PreFunc(hideUnlessClipmap)
        .Options(FloatSliderOptions()
                     .Tooltip("Meia-extensão do nível mais interno, em unidades de mundo. Toda a escada "
                              "segue: o nível i é este valor vezes 2^i.\n\n"
                              "120 com oito níveis alcança cerca de 15000 -- duas vezes e meia a escada de "
                              "cascatas -- com o nível 0 medindo 240 unidades de lado.\n\n"
                              "Menor deixa a sombra mais nítida aos pés e encurta o alcance total; maior faz "
                              "o contrário.")
                     .Min(20.0f)   // SHADOW_MAP_MIN_CLIPMAP_BASE
                     .Max(2000.0f) // SHADOW_MAP_MAX_CLIPMAP_BASE
                     .Step(10.0f)
                     .DefaultValue(120.0f) // SHADOW_MAP_DEFAULT_CLIPMAP_BASE
                     .Format("%.0f"));

    // ===========================================================================================
    // Technique 4 -- cascade split ladder.
    // ===========================================================================================
    path = { "Sombras", "Forma do Mapa", SECTION_COLUMN_1 };

    auto hideUnlessAutoLadder = [](WidgetInfo& info) {
        info.isHidden = ShadowAdvancedOff() || CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderMode"),
                                                              0) != 1; // SHADOW_MAP_LADDER_PRACTICAL
    };

    AddWidget(path, "Distribuição das Faixas", WIDGET_CVAR_COMBOBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderMode"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(ComboboxOptions()
                     .ComboMap(shadowLadderModeLabels)
                     .DefaultIndex(0) // SHADOW_MAP_DEFAULT_LADDER_MODE
                     .Tooltip("Onde caem as fronteiras entre as cascatas, o que decide o tamanho do texel "
                              "em cada faixa -- ou seja, o tamanho do degrau ANTES de qualquer filtro.\n\n"
                              "Manual usa os sliders de distância da aba Wind Waker Style, como sempre.\n\n"
                              "Automática recalcula as fronteiras internas para uniformizar o texel entre as "
                              "faixas. O ALCANCE não muda: a última distância continua sendo a sua.\n\n"
                              "Não custa nada -- são três números, sem shader e sem memória nova."));
    AddWidget(path, "Uniforme <-> Logarítmica", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderLambda"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAutoLadder)
        .Options(FloatSliderOptions()
                     .Tooltip("0% = faixas com a mesma PROFUNDIDADE. 100% = faixas com a mesma RAZÃO, que é "
                              "o que deixa o texel do mesmo tamanho em todas.\n\n"
                              "Logarítmica pura joga a primeira fronteira muito perto da câmera, e uma "
                              "cascata que cobre quase nada desperdiça uma fatia inteira. A mistura é o "
                              "meio-termo usual.")
                     .Min(0.0f)
                     .Max(1.0f)
                     .Step(0.01f)
                     .DefaultValue(0.85f) // SHADOW_MAP_DEFAULT_LADDER_LAMBDA
                     .IsPercentage());
    AddWidget(path, "Distância Inicial: %.0f", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.LadderNear"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAutoLadder)
        .Options(FloatSliderOptions()
                     .Tooltip("De onde a escada é gerada.\n\n"
                              "Não é o plano próximo da câmera, que é pequeno o bastante para puxar a "
                              "primeira fronteira para quase zero. É a distância a partir da qual vale a "
                              "pena resolver sombras com precisão.")
                     .Min(1.0f)
                     .Max(1000.0f)
                     .Step(5.0f)
                     .DefaultValue(40.0f) // SHADOW_MAP_DEFAULT_LADDER_NEAR
                     .Format("%.0f"));

    AddWidget(path, "Resolução por Nível", WIDGET_CVAR_COMBOBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowQuality.ClipmapResolution"))
        .RaceDisable(false)
        .PreFunc(hideUnlessClipmap)
        .Options(ComboboxOptions()
                     .ComboMap(shadowMapResolutionLabels)
                     .DefaultIndex(4096) // SHADOW_MAP_DEFAULT_CLIPMAP_RESOLUTION
                     .Tooltip("Resolução de cada nível, SEPARADA da Resolução das cascatas.\n\n"
                              "Separada porque as duas querem coisas opostas: um clipmap quer muitos níveis "
                              "pequenos, uma escada de cascatas quer poucos grandes. Compartilhar o número faz um "
                              "passar fome ou o outro estourar a memória.\n\n"
                              "Unidades de mundo por texel, contra a escada de cascatas na SUA 4096:\n"
                              "  distância    cascatas   clip 2048   clip 4096\n"
                              "       150        0,090       0,234       0,117\n"
                              "       900        0,740       0,938       0,469\n"
                              "      2000        0,740       3,750       1,875\n"
                              "      4000        3,600       7,500       3,750\n\n"
                              "A 2048 o clipmap perde em quase toda distância. Ele precisa da resolução maior por "
                              "um motivo estrutural: um nível é um QUADRADO CENTRADO NA CÂMERA, enquanto uma "
                              "cascata é uma laje AJUSTADA AO CAMPO DE VISÃO. A câmera olha para um lado só, "
                              "então o quadrado gasta a maior parte da área em chão que ninguém está vendo. É o "
                              "preço das fronteiras que andam com você.\n\n"
                              "8 níveis a 4096 = 256 MB. A camada de personagens usa só os dois níveis mais "
                              "internos, então 4096 lá é desperdício -- baixar Resolução (Personagens) para 1024 "
                              "tira 60 MB do total.\n\n"
                              "Subir aqui deixa tudo mais nítido de uma vez; subir os Níveis estende o alcance."));

    // What the ladder ACTUALLY produced, read back from the renderer rather than recomputed here.
    //
    // This is the answer to "what did automatic decide", and it cannot be got any other way: with the
    // automatic ladder on, the split sliders no longer say where the bands are, and the texel size never
    // did -- it falls out of the projection the fit builds from the camera. Shown in both modes, so manual
    // and automatic can be compared against the same numbers.
    //
    // WIDGET_TEXT draws widget.name and PreFunc runs first, so rewriting the name each frame is what makes
    // it live.
    AddWidget(path, "Resultado", WIDGET_SEPARATOR_TEXT).PreFunc(advOnly);
    AddWidget(path, "Cascatas em uso:", WIDGET_TEXT).RaceDisable(false).PreFunc([](WidgetInfo& info) {
        info.isHidden = ShadowAdvancedOff();
        if (!info.isHidden) {
            const char* report = ToonLighting_ShadowMapCascadeReport();
            info.name = std::string("Cascatas em uso:\n") + (report != nullptr ? report : "");
        }
    });

    // The cross-fade between cascades. It has always existed and has never been reachable from the menu --
    // only from the console -- which is why a hard cascade seam had no control to soften it.
    AddWidget(path, "Suavidade da Transição", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowMap.BlendFraction"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(FloatSliderOptions()
                     .Tooltip("Quanto do alcance de cada faixa é usado para dissolver na faixa seguinte.\n\n"
                              "O shader amostra as DUAS cascatas dentro dessa banda e mistura com "
                              "smoothstep. É isso que impede que a mudança de resolução apareça como uma "
                              "linha dura varrendo o chão conforme a câmera anda.\n\n"
                              "Em 0% a transição é um corte seco. Valores altos suavizam mais, mas custam: "
                              "dentro da banda cada pixel faz o dobro das leituras.\n\n"
                              "A banda resultante de cada faixa aparece em 'Cascatas em uso', acima.")
                     .Min(0.0f)
                     .Max(0.5f)
                     .Step(0.01f)
                     .DefaultValue(0.2f) // SHADOW_MAP_DEFAULT_BLEND_FRACTION
                     .IsPercentage());

    // Every control below the master switch is hidden while it is off, so the page shows what is in play --
    // and, like everything else in here, only while the advanced switch is on.
    auto hideUnlessAcne = [](WidgetInfo& info) {
        info.isHidden = ShadowAdvancedOff() || !CVarGetInteger(CVAR_ENHANCEMENT("Graphics.ShadowAcne.Enabled"), 1);
    };
    path = { "Sombras", "Acne", SECTION_COLUMN_1 };
    AddSidebarEntry("Sombras", path.sidebarName, 3);

    AddWidget(path, "O que é acne", WIDGET_SEPARATOR_TEXT).PreFunc(advOnly);
    AddWidget(path, "Uma superfície fazendo sombra em si mesma: listras escuras no chão que, num", WIDGET_TEXT)
        .PreFunc(advOnly);
    AddWidget(path, "ângulo rasante, viram raios convergindo no horizonte.", WIDGET_TEXT).PreFunc(advOnly);
    AddWidget(path, "A defesa padrão é o slope bias do rasterizador, aplicado enquanto o mapa de", WIDGET_TEXT)
        .PreFunc(advOnly);
    AddWidget(path, "profundidade é escrito, e normalmente ela basta. Estes métodos são para onde não", WIDGET_TEXT)
        .PreFunc(advOnly);
    AddWidget(path, "basta: um texel muito grande, ou uma superfície quase de lado para a luz.", WIDGET_TEXT)
        .PreFunc(advOnly);

    AddWidget(path, "Ativar Correção", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowAcne.Enabled"))
        .RaceDisable(false)
        .PreFunc(advOnly)
        .Options(CheckboxOptions().DefaultValue(true).Tooltip(
            "Chave geral dos métodos abaixo.\n\n"
            "LIGADA por padrão, com os valores que se mostraram necessários na prática: um deslocamento "
            "pequeno pela normal, escalado pela inclinação, e mais nada.\n\n"
            "O slope bias do rasterizador continua sendo a defesa principal. Estes métodos cobrem onde ele "
            "não basta."));

    AddWidget(path, "Restaurar Tudo ao Padrão", WIDGET_BUTTON)
        .PreFunc(hideUnlessAcne)
        .Callback([](WidgetInfo& info) {
            CVarClear(CVAR_ENHANCEMENT("Graphics.ShadowAcne.Enabled"));
            CVarClear(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalOffset"));
            CVarClear(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalTexels"));
            CVarClear(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeScaled"));
            CVarClear(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeMax"));
            Ship::Context::GetInstance()->GetWindow()->GetGui()->SaveConsoleVariablesNextFrame();
        })
        .Options(ButtonOptions().Tooltip("Devolve os métodos abaixo à combinação recomendada."));

    // ===========================================================================================
    // Method 1 -- normal offset. The one that is correct in principle.
    // ===========================================================================================
    AddWidget(path, "Método 1: Deslocamento pela Normal", WIDGET_SEPARATOR_TEXT).PreFunc(hideUnlessAcne);
    AddWidget(path, "Ativar Deslocamento pela Normal", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalOffset"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAcne)
        .Options(CheckboxOptions().DefaultValue(true).Tooltip(
            "Afasta o ponto de amostragem da superfície ao longo da NORMAL dela, antes de projetar.\n\n"
            "É o único método correto em princípio, e não um remendo: o erro que se está corrigindo é um "
            "deslocamento no espaço do mundo, e este também é.\n\n"
            "Como o deslocamento é ao longo da superfície e não da luz, ele NÃO descola a sombra do pé de "
            "quem a projeta -- que é o preço dos outros dois métodos.\n\n"
            "Comece por aqui. Na prática este e a Escala por Inclinação resolvem sozinhos."));
    AddWidget(path, "Distância: %.2f texels", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowAcne.NormalTexels"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAcne)
        .Options(FloatSliderOptions()
                     .Tooltip("Em múltiplos do texel da cascata amostrada, não em unidades de mundo.\n\n"
                              "O erro que isto corrige é ele próprio do tamanho de um texel, então uma "
                              "distância fixa em unidades de mundo seria grande demais na cascata próxima "
                              "e pequena demais na distante.\n\n"
                              "Se ainda houver listras, aumente. Se as sombras começarem a encolher perto "
                              "dos contatos, diminua.")
                     .Min(0.0f)
                     .Max(8.0f) // SHADOW_MAP_MAX_ACNE_NORMAL_TEXELS
                     .Step(0.1f)
                     .DefaultValue(0.6f) // SHADOW_MAP_DEFAULT_ACNE_NORMAL_TEXELS
                     .Format("%.2f"));

    // ===========================================================================================
    // Method 2 -- slope scaling. Not an offset of its own; it shapes the other three.
    // ===========================================================================================
    AddWidget(path, "Método 2: Escala por Inclinação", WIDGET_SEPARATOR_TEXT).PreFunc(hideUnlessAcne);
    AddWidget(path, "Escalar pela Inclinação", WIDGET_CVAR_CHECKBOX)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeScaled"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAcne)
        .Options(CheckboxOptions().DefaultValue(true).Tooltip(
            "Não é um deslocamento próprio: multiplica o deslocamento pela normal por quão de lado a "
            "superfície está "
            "em relação à luz.\n\n"
            "Acne é um problema de ângulo rasante. Uma superfície de frente para a luz não tem nenhum; uma "
            "de lado tem a profundidade disparando ao longo de um texel. Escalar pelo ângulo gasta a "
            "correção onde ela é necessária e quase nada onde não é."));
    AddWidget(path, "Limite da Escala: %.1fx", WIDGET_CVAR_SLIDER_FLOAT)
        .CVar(CVAR_ENHANCEMENT("Graphics.ShadowAcne.SlopeMax"))
        .RaceDisable(false)
        .PreFunc(hideUnlessAcne)
        .Options(FloatSliderOptions()
                     .Tooltip("Teto da multiplicação, e ele NÃO é opcional.\n\n"
                              "O fator cresce para infinito conforme a superfície fica de lado para a luz, "
                              "e um deslocamento sem limite ali joga a amostra para outra parte da cena "
                              "inteira.")
                     .Min(1.0f)
                     .Max(10.0f) // SHADOW_MAP_MAX_ACNE_SLOPE_MAX
                     .Step(0.1f)
                     .DefaultValue(3.5f) // SHADOW_MAP_DEFAULT_ACNE_SLOPE_MAX
                     .Format("%.1f"));
}
} // namespace SohGui
