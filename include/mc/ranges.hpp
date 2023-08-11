#pragma once

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 202110L

#include <ranges>
namespace __ranges = std::ranges;

#else
static_assert(false);

#endif
