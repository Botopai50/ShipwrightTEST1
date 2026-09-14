#ifndef SOHMENU_H
#define SOHMENU_H

#include <libultraship/libultraship.h>
#include "Menu.h"
#include <fast/backends/gfx_rendering_api.h>
#include "soh/cvar_prefixes.h"

extern "C" {
#include "z64.h"
}

#ifdef __cplusplus
extern "C" {
#endif
void enableBetaQuest();
void disableBetaQuest();
#ifdef __cplusplus
}
#endif

namespace SohGui {
static std::map<int32_t, const char*> languages = {
    { LANGUAGE_ENG, "English" },
    { LANGUAGE_GER, "German" },
    { LANGUAGE_FRA, "French" },
    { LANGUAGE_JPN, "Japanese" },
};
// Offered shadow-map sizes, shared by every panel that sizes one: the cascade ladder's, the actor layer's
// and the clipmap's. Keys are the actual resolution, which is what the CVar stores, so a combobox reads and
// writes the value the renderer uses rather than an index into this list.
//
// Lives here rather than in one of the menu files because two of them need it, and a `static` copy in each
// is two lists to keep in step.
//
// It stops at 4096 (see SHADOW_MAP_MAX_RESOLUTION). 8192 was offered briefly and taken back out: a slice
// quadruples with each step, so it is 128 MB apiece and puts a clipmap over a gigabyte. Sharper shadows
// come from more clipmap levels, which cost linearly.
inline const std::map<int32_t, const char*> shadowMapResolutionLabels = {
    { 512, "512" },
    { 1024, "1024" },
    { 2048, "2048" },
    { 4096, "4096" },
};

void UpdateMenuTricks();
void UpdateMenuLocations();

class SohMenu : public Ship::Menu {
  public:
    SohMenu(const std::string& consoleVariable, const std::string& name);

    void InitElement() override;
    void DrawElement() override;
    void UpdateElement() override;
    void Draw() override;

    void AddSidebarEntry(std::string sectionName, std::string sidbarName, uint32_t columnCount);
    WidgetInfo& AddWidget(WidgetPath& pathInfo, std::string widgetName, WidgetType widgetType);
    void AddMenuElements();
    void AddMenuSettings();
    void AddMenuEnhancements();
    void AddMenuDevTools();
    void AddMenuRandomizer();
    void AddMenuNetwork();
    void AddMenuWindWakerStyle();
    void AddMenuShadows();
    static void UpdateLanguageMap(std::map<int32_t, const char*>& languageMap);

  private:
    char mGitCommitHashTruncated[8];
    bool mIsTaggedVersion;
    bool mMenuElementsInitialized = false;
};
} // namespace SohGui

#endif // SOHMENU_H
