#include "FpsOverlayWindow.h"
#include "soh/OTRGlobals.h"

void FpsOverlayWindow::Draw() {
    if (!IsVisible()) {
        return;
    }

    // Pinned to the top-left of the viewport's WORK area, which is the part below whatever menu bar is
    // showing -- so the read-out does not slide under the bar when it appears and reappear when it goes.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float margin = 10.0f;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + margin, viewport->WorkPos.y + margin), ImGuiCond_Always);
    // The background. Set through ImGui rather than by drawing a rectangle so it follows the theme's window
    // colour and only the alpha is ours.
    ImGui::SetNextWindowBgAlpha(0.35f);

    // NoInputs is the one that matters: without it an invisible rectangle in the corner of the screen eats
    // clicks meant for the game or the menu behind it. NoSavedSettings keeps imgui.ini from remembering a
    // position that SetNextWindowPos is going to override every frame anyway.
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;

    if (ImGui::Begin("##FpsOverlay", nullptr, flags)) {
        DrawElement();
    }
    ImGui::End();

    // The window has no close button of its own, so this only ever carries a change made from the menu --
    // but it has to run for the CVar and the visibility to agree.
    SyncVisibilityConsoleVariable();
}

void FpsOverlayWindow::DrawElement() {
    const ImGuiIO& io = ImGui::GetIO();

    // ImGui's own running average rather than a counter of our own. It is already smoothed over the last
    // second, which is what makes the number readable -- an instantaneous frame time flickers too fast to
    // read and tells you about one frame rather than about how the game is running.
    const float framerate = io.Framerate;
    const float msPerFrame = framerate > 0.0f ? (1000.0f / framerate) : 0.0f;

    ImGui::PushFont(OTRGlobals::Instance->fontMonoLarger);
    ImGui::Text("%.0f FPS", framerate);
    // The millisecond figure is the one to read when comparing builds: frames per second is a reciprocal, so
    // equal drops in it mean very different amounts of work at 30 than at 120.
    ImGui::Text("%.2f ms", msPerFrame);
    ImGui::PopFont();
}
