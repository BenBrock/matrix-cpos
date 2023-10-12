#pragma once

#include <mc/cpos.hpp>
#include <mc/ranges.hpp>

namespace mc {

template <__ranges::random_access_range C, row_iterable A,
          __ranges::random_access_range B>
void spmv(C&& c, A&& a, B&& b) {
  for (auto&& [i, row] : rows(a)) {
    for (auto&& [j, a_v] : row) {
      c[i] += a_v * b[j];
    }
  }
}

template <__ranges::random_access_range C, column_iterable A,
          __ranges::random_access_range B>
  requires(row_iterable<A>)
void spmv(C&& c, A&& a, B&& b) {
  for (auto&& [j, column] : columns(a)) {
    for (auto&& [i, a_v] : column) {
      c[i] += a_v * b[j];
    }
  }
}

} // namespace mc
