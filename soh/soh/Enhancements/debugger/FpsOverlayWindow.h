#ifndef SOH_FPS_OVERLAY_H
#define SOH_FPS_OVERLAY_H

#include <libultraship/libultraship.h>

// A fixed read-out in the corner of the screen, for watching the frame time while playing.
//
// Separate from the Stats window rather than a mode of it: that one is a normal window the player moves,
// resizes and focuses, which is what you want when reading it and exactly what you do not want from
// something meant to sit over the game while both hands are on the controller.
class FpsOverlayWindow final : public Ship::GuiWindow {
  public:
    using GuiWindow::GuiWindow;
    ~FpsOverlayWindow() override = default;

  protected:
    void InitElement() override {
    }
    void UpdateElement() override {
    }
    void DrawElement() override;
    // Placed and styled here rather than left to the base class, which opens a movable window at whatever
    // position ImGui remembers. This one is pinned, click-through and unsaved.
    void Draw() override;
};

#endif // SOH_FPS_OVERLAY_H
