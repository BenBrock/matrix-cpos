#pragma once

#include <cassert>
#include <mc/cpos.hpp>
#include <mc/ranges.hpp>

#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
#endif

namespace mc {

#ifdef SYCL_LANGUAGE_VERSION
template <__ranges::contiguous_range C, row_iterable A,
          __ranges::contiguous_range B>
sycl::event spmv(sycl::queue& q, C&& c, A&& a, B&& b) {
  std::size_t block_size = 128;
  std::size_t num_rows = __ranges::distance(mc::rows(a));
  sycl::nd_range<1> range(num_rows * block_size, block_size);

  std::span c_(c);
  std::span b_(b);
  auto a_ = a;

  auto rows = mc::rows(a);

  using T = __ranges::range_value_t<C>;

  return q.parallel_for(range, [=](auto idx) {
    auto local_id = idx.get_local_id(0);
    auto group_id = idx.get_group(0);
    auto group_size = idx.get_group_range(0);

    // auto&& rows = mc::rows(a_);
    auto num_rows = __ranges::size(rows);

    if (group_id < num_rows) {

      auto&& [i, row] = rows[group_id];

      sycl::atomic_ref<T, sycl::memory_order::relaxed,
                       sycl::memory_scope::work_group>
          c_v(c_[i]);

      for (auto j_ptr = local_id; j_ptr < __ranges::size(row);
           j_ptr += group_size) {
        auto&& [j, a_v] = row[j_ptr];

        c_v += a_v * b_[j];
      }
    }
  });
}

#endif

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
