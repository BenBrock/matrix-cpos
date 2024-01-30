#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <mc/ranges.hpp>

namespace mc {

template <typename T = float, typename I = int,
          typename Allocator = std::allocator<T>>
auto generate_vec(I n, std::size_t nnz, std::size_t seed = 0,
                  const Allocator& alloc = Allocator{}) {
  using IAllocator = std::allocator_traits<Allocator>::template rebind_alloc<I>;
  IAllocator i_alloc(alloc);

  std::vector<T, Allocator> values(alloc);

  values.reserve(nnz);

  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, n - 1);
  std::uniform_real_distribution d_f(0.0, 100.0);
  std::uniform_int_distribution<I> d_m(0, nnz);

  for (std::size_t i = 0; i < nnz; i++) {
    values.push_back(d_f(g));
  }

  return std::tuple(values, n, I(nnz));
}

} // namespace mc
