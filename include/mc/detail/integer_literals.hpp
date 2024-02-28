#pragma once

#include <string>

namespace mc {

template <size_t I>
struct integer {};

} // namespace mc

template <char... Chars>
constexpr auto operator""_i() {
  return mc::integer<[] {
    size_t rv = 0;
    for (auto c : std::string{Chars...})
      rv = rv * 10 + (c - '0');
    return rv;
  }()>{};
}
