#pragma once

#include <mc/detail/integer_literals.hpp>
#include <mc/ranges.hpp>

namespace mc {

namespace __detail {
template <std::size_t I, std::size_t N, typename Fn, typename Iter>
constexpr void unroll_impl_(Fn&& fn, Iter&& p) {
  fn(*(p + I));
  if constexpr (I + 1 < N) {
    unroll_impl_<I + 1, N>(fn, p);
  }
}
} // namespace __detail

template <std::size_t N, __ranges::random_access_range R, typename Fn>
constexpr void unrolled_for_each(integer<N> unroll_factor, R&& r, Fn&& fn) {
  std::size_t strip_mined_size = (__ranges::size(r) / N) * N;

  auto p = __ranges::begin(r);
  std::size_t i;
  for (i = 0; i < strip_mined_size; i += N) {
    auto section_p = p + i;
    __detail::unroll_impl_<0, N>(fn, section_p);
  }

  for (; i < __ranges::size(r); i++) {
    fn(*(p + i));
  }
}

} // namespace mc
