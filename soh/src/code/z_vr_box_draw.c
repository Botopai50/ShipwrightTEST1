#include "global.h"

#include "soh/frame_interpolation.h"
#include "soh/Enhancements/Graphics/ToonLighting.h"

Mtx* sSkyboxDrawMatrix;

Mtx* SkyboxDraw_UpdateMatrix(SkyboxContext* skyboxCtx, f32 x, f32 y, f32 z) {
    Matrix_Translate(x, y, z, MTXMODE_NEW);
    Matrix_Scale(1.0f, 1.0f, 1.0f, MTXMODE_APPLY);
    Matrix_RotateX(skyboxCtx->rot.x, MTXMODE_APPLY);
    Matrix_RotateY(skyboxCtx->rot.y, MTXMODE_APPLY);
    Matrix_RotateZ(skyboxCtx->rot.z, MTXMODE_APPLY);
    return MATRIX_TOMTX(sSkyboxDrawMatrix);
}

void SkyboxDraw_Draw(SkyboxContext* skyboxCtx, GraphicsContext* gfxCtx, s16 skyboxId, s16 blend, f32 x, f32 y, f32 z) {
    OPEN_DISPS(gfxCtx);
    FrameInterpolation_RecordOpenChild(NULL, FrameInterpolation_GetCameraEpoch());

    // SOH [Enhancement] Cascaded shadow maps: the sky must not receive shadow. It is real 3D geometry, so it
    // was being projected into the cascades like any other surface and came back with dark streaks smeared
    // across it. Turned back on at the end of the function so nothing after it inherits the state.
    s32 shadowMapOn = ToonLighting_ShadowMapEnabled();
    if (shadowMapOn) {
        gSPShadowMapReceiveOff(POLY_OPA_DISP++);
    }

    // SOH [Enhancement] Mipmapping: the sky reads its textures at full size. Turned back off at the end of
    // the function so nothing after it inherits the state, exactly like the marker above.
    //
    // The sky is drawn as separate faces, each holding its own texture, laid edge to edge. At the top level
    // the two sides of a join disagree by less than a texel and nothing sees it. Each smaller copy averages
    // twice as many texels together, and each side averages only its own, so the disagreement widens with
    // every level until it reads as a line drawn along the join -- three of them meeting overhead, which is
    // the shape a cube's corner makes from the inside.
    //
    // The renderer already does this for the interface, which it recognises by the projection: a draw with
    // no perspective term is on-screen furniture. That test cannot reach here, because the sky is genuinely
    // 3D and genuinely in perspective. Nothing in the render state says "this face continues into the next
    // texture" either -- it is knowledge about the ART, and the game is the only thing that has it.
    gSPTextureLodClampOn(POLY_OPA_DISP++);

    Gfx_SetupDL_40Opa(gfxCtx);

    // gsSPShaderTest(POLY_OPA_DISP++);

    /*gSPSegment(POLY_OPA_DISP++, 0x7, skyboxCtx->staticSegments[0]);
    gSPSegment(POLY_OPA_DISP++, 0x8, skyboxCtx->staticSegments[1]);
    gSPSegment(POLY_OPA_DISP++, 0x9, skyboxCtx->palettes);*/

    gDPSetPrimColor(POLY_OPA_DISP++, 0x00, 0x00, 0, 0, 0, blend);
    gSPTexture(POLY_OPA_DISP++, 0x8000, 0x8000, 0, G_TX_RENDERTILE, G_ON);

    sSkyboxDrawMatrix = Graph_Alloc(gfxCtx, sizeof(Mtx));

    Matrix_Translate(x, y, z, MTXMODE_NEW);
    Matrix_Scale(1.0f, 1.0f, 1.0f, MTXMODE_APPLY);
    Matrix_RotateX(skyboxCtx->rot.x, MTXMODE_APPLY);
    Matrix_RotateY(skyboxCtx->rot.y, MTXMODE_APPLY);
    Matrix_RotateZ(skyboxCtx->rot.z, MTXMODE_APPLY);
    MATRIX_TOMTX(sSkyboxDrawMatrix);
    gSPMatrix(POLY_OPA_DISP++, sSkyboxDrawMatrix, G_MTX_NOPUSH | G_MTX_LOAD | G_MTX_MODELVIEW);

    gDPSetColorDither(POLY_OPA_DISP++, G_CD_MAGICSQ);
    gDPSetTextureFilter(POLY_OPA_DISP++, G_TF_BILERP);

    if (skyboxCtx->palette_size == 256) {
        gDPLoadTLUT_pal256(POLY_OPA_DISP++, skyboxCtx->palettes[0]);
    } else {
        gDPLoadTLUT_pal128(POLY_OPA_DISP++, 0, skyboxCtx->palettes[0]);
        gDPLoadTLUT_pal128(POLY_OPA_DISP++, 1, skyboxCtx->palettes[1]);
    }
    gDPSetTextureLUT(POLY_OPA_DISP++, G_TT_RGBA16);
    gDPSetTextureConvert(POLY_OPA_DISP++, G_TC_FILT);

    if (skyboxCtx->unk_140) {
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[0]);
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[1]);

        gDPPipeSync(POLY_OPA_DISP++);
        gDPLoadTLUT_pal256(POLY_OPA_DISP++, skyboxCtx->palettes[1]);
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[2]);
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[3]);

        //! @bug All shops only provide 2 faces for their sky box. Mask shop is missing from the condition
        // meaning that the Mask shop will render the previously loaded sky box values, or uninitialized data.
        // This effect is not noticed as the faces are behind the camera, but will cause a crash in SoH.
        // SOH General: We have added the Mask shop to this check so only the 2 expected faces are rendered.
        if (skyboxId != SKYBOX_BAZAAR && skyboxId != SKYBOX_HAPPY_MASK_SHOP) {
            if (skyboxId < SKYBOX_KOKIRI_SHOP || skyboxId > SKYBOX_BOMBCHU_SHOP) {
                // Skip remaining faces for most shop skyboxes

                gDPPipeSync(POLY_OPA_DISP++);
                gDPLoadTLUT_pal256(POLY_OPA_DISP++, skyboxCtx->palettes[2]);
                gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[4]);
                gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[5]);

                gDPPipeSync(POLY_OPA_DISP++);
                if (skyboxCtx->unk_140 != 2) {
                    gDPLoadTLUT_pal256(POLY_OPA_DISP++, skyboxCtx->palettes[3]);
                    gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[6]);
                    gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[7]);
                }
            }
        }
    } else {
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[0]);
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[2]);
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[4]);
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[6]);
        gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[8]);
        if (skyboxId == SKYBOX_CUTSCENE_MAP) {
            gSPDisplayList(POLY_OPA_DISP++, skyboxCtx->dListBuf[10]);
        }
    }

    gDPPipeSync(POLY_OPA_DISP++);
    // gsSPShaderTest2(POLY_OPA_DISP++);

    if (shadowMapOn) {
        gSPShadowMapReceiveOn(POLY_OPA_DISP++);
    }
    gSPTextureLodClampOff(POLY_OPA_DISP++);

    FrameInterpolation_RecordCloseChild();
    CLOSE_DISPS(gfxCtx);
}

void SkyboxDraw_Update(SkyboxContext* skyboxCtx) {
}
