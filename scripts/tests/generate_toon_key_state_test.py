"""Generate a compile check from the actual toon state declaration.

Run from the repository root, after initializing libultraship:
  python scripts/tests/generate_toon_key_state_test.py build-tests/toon_key_state.cpp
  cl /nologo /std:c++20 /permissive- /Zc:preprocessor /c /Ilibultraship/include build-tests/toon_key_state.cpp

With Clang, use -std=c++20 -Werror=non-c-typedef-for-linkage -fsyntax-only.
MSVC rejects default member initializers inside an anonymous typedef class
(C7626). Extracting the real declaration, instead of a simplified mock, catches
that failure while also checking the value initialization used by try_emplace.
"""

import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / "soh/soh/Enhancements/Graphics/ToonLighting.cpp").read_text(encoding="utf-8")
    declarations = re.findall(
        r"(?:typedef struct \{|struct ToonKeyState \{)\n    f32 dir\[3\];.*?\n\}(?: ToonKeyState)?;",
        source,
        re.DOTALL,
    )
    if len(declarations) != 1:
        raise RuntimeError("Expected exactly one ToonKeyState declaration; update the extraction pattern")

    test = """#include <type_traits>
#include "fast/toon_shading.h"
using f32 = float;
using u8 = unsigned char;
struct LightInfo;
""" + declarations[0] + """
static_assert(std::is_aggregate_v<ToonKeyState>);
constexpr ToonKeyState initial{};
static_assert(!initial.multipleLights);
static_assert(initial.dir[0] == 0 && initial.colVel[2] == 0);
static_assert(initial.shadowScale == 0 && initial.shadowScaleVel == 0);
static_assert(initial.floorValid == 0 && initial.floorSampled == 0);
constexpr bool emptySources() {
    for (const auto* source : initial.localSources) {
        if (source != nullptr) return false;
    }
    return true;
}
static_assert(emptySources());
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(test, encoding="utf-8")


if __name__ == "__main__":
    main()
