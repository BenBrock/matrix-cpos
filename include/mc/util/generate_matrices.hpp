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
auto generate_csr(I m, I n, std::size_t nnz, std::size_t seed = 0,
                  const Allocator& alloc = Allocator{}) {
  using IAllocator = std::allocator_traits<Allocator>::template rebind_alloc<I>;
  IAllocator i_alloc(alloc);

  std::vector<T, Allocator> values(alloc);
  std::vector<I, IAllocator> colind(i_alloc);

  values.reserve(nnz);
  colind.reserve(nnz);

  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, n - 1);
  std::uniform_real_distribution d_f(0.0, 100.0);
  std::uniform_int_distribution<I> d_m(0, nnz);

  for (std::size_t i = 0; i < nnz; i++) {
    colind.push_back(d(g));
  }

  for (std::size_t i = 0; i < nnz; i++) {
    values.push_back(d_f(g));
  }

  std::vector<I, IAllocator> rowptr(i_alloc);
  rowptr.reserve(m + 1);
  rowptr.push_back(0);
  for (std::size_t i = 0; i < m - 1; i++) {
    rowptr.push_back(d_m(g));
  }
  rowptr.push_back(nnz);

  std::sort(rowptr.begin(), rowptr.end());

  for (std::size_t i = m; i >= 1; i--) {
    rowptr[i] -= rowptr[i - 1];
  }

  std::inclusive_scan(rowptr.begin(), rowptr.end(), rowptr.begin());

  return std::tuple(values, rowptr, colind, mc::index<I>(m, n), I(nnz));
}

template <typename T = float, typename I = int>
auto generate_csc(I m, I n, std::size_t nnz, std::size_t seed = 0) {
  auto&& [values, colptr, rowind, shape_, nnz_] =
      generate_csr<T, I>(n, m, nnz, seed);
  return std::tuple(std::move(values), std::move(colptr), std::move(rowind),
                    mc::index<I>(m, n), I(nnz));
}

template <typename T = float, typename I = int,
          typename Allocator = std::allocator<T>>
auto generate_dcsr(I m, I n, std::size_t nnz, std::size_t seed = 0,
                   const Allocator& alloc = Allocator{}) {
  using IAllocator = std::allocator_traits<Allocator>::template rebind_alloc<I>;
  IAllocator i_alloc(alloc);

  auto&& [values, rowptr_, colind, shape, _] =
      generate_csr<T, I, Allocator>(m, n, nnz, seed, alloc);

  std::size_t num_rows = 0;

  for (std::size_t i = 0; i < m; i++) {
    if (rowptr_[i + 1] - rowptr_[i] > 0) {
      num_rows++;
    }
  }

  std::vector<I, IAllocator> rowptr(num_rows + 1, i_alloc);
  std::vector<I, IAllocator> rowind(num_rows, i_alloc);

  I rp_idx = 0;
  for (std::size_t i = 0; i < m; i++) {
    if (rowptr_[i + 1] - rowptr_[i] > 0) {
      rowptr[rp_idx] = rowptr_[i];
      rowind[rp_idx] = i;
      rp_idx++;
    }
  }
  rowptr[num_rows] = nnz;

  return std::tuple(values, rowind, rowptr, colind, shape, num_rows, nnz);
}

template <typename T = float, typename I = int>
auto generate_coo(I m, I n, std::size_t nnz, std::size_t seed = 0) {
  std::vector<T> values;
  std::vector<I> rowind;
  std::vector<I> colind;

  values.reserve(nnz);
  rowind.reserve(nnz);
  colind.reserve(nnz);

  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d_row(0, m - 1);
  std::uniform_int_distribution<I> d_col(0, n - 1);
  std::uniform_real_distribution d_val(0.0, 100.0);

  for (std::size_t i = 0; i < nnz; i++) {
    values.push_back(d_val(g));
  }

  for (std::size_t i = 0; i < nnz; i++) {
    rowind.push_back(d_row(g));
  }

  for (std::size_t i = 0; i < nnz; i++) {
    colind.push_back(d_col(g));
  }

  auto indices_view = __ranges::views::zip(rowind, colind);

  auto values_view = __ranges::views::zip(indices_view, values);

  auto sort_fn = [](auto&& x, auto&& y) {
    auto&& [x_idx, x_v] = x;
    auto&& [y_idx, y_v] = y;
    auto&& [x_i, x_j] = x_idx;
    auto&& [y_i, y_j] = y_idx;

    if (x_i == y_i) {
      return x_j < y_j;
    } else {
      return x_i < y_i;
    }
  };

  std::sort(values_view.begin(), values_view.end(), sort_fn);

  return std::tuple(values, rowind, colind, mc::index<I>(m, n), I(nnz));
}

template <typename T = float>
auto generate_dense(std::size_t m, std::size_t n, std::size_t seed = 0) {
  std::vector<T> v;
  v.reserve(m * n);

  std::mt19937 g(seed);
  std::uniform_real_distribution d(0.0, 100.0);

  for (std::size_t i = 0; i < m * n; i++) {
    v.push_back(d(g));
  }

  return std::tuple(v, mc::index(m, n));
}

template <typename T = float, typename I = int>
auto get_dcsr(I m, I n, std::size_t nnz, std::size_t seed = 0) {
  auto&& [values, rowptr_, colind, shape, _] =
      generate_csr<T, I>(m, n, nnz, seed);

  std::size_t num_rows = 0;

  for (std::size_t i = 0; i < m; i++) {
    if (rowptr_[i + 1] - rowptr_[i] > 0) {
      num_rows++;
    }
  }

  std::vector<I> rowptr(num_rows + 1);
  std::vector<I> rowind(num_rows);

  I rp_idx = 0;
  for (std::size_t i = 0; i < m; i++) {
    if (rowptr_[i + 1] - rowptr_[i] > 0) {
      rowptr[rp_idx] = rowptr_[i];
      rowind[rp_idx] = i;
      rp_idx++;
    }
  }
  rowptr[num_rows] = nnz;

  auto* values_p = new std::vector(values);
  auto* rowptr_p = new std::vector(rowptr);
  auto* rowind_p = new std::vector(rowind);
  auto* colind_p = new std::vector(colind);

  return mc::dcsr_matrix_view(values_p->begin(), rowind_p->begin(),
                              rowptr_p->begin(), colind_p->begin(), shape,
                              num_rows, nnz);
}

template <typename T = float>
auto get_dense(std::size_t m, std::size_t n, std::size_t seed = 0) {
  auto* v_ptr = new std::vector<T>(m * n);
  auto& v = *v_ptr;

  std::mt19937 g(seed);
  std::uniform_real_distribution d(0.0, 100.0);

  for (auto&& x : v) {
    x = d(g);
  }

  return mc::dense_matrix_view(v.begin(), {m, n});
}

template <typename T = float, typename I = int>
auto get_csr(I m, I n, std::size_t nnz, std::size_t seed = 0) {
  auto [values_, rowptr_, colind_, shape, _] =
      mc::generate_csr<T, I>(m, n, nnz, seed);

  auto* values = new std::vector(values_);
  auto* rowptr = new std::vector(rowptr_);
  auto* colind = new std::vector(colind_);

  mc::csr_matrix_view view(values->begin(), rowptr->begin(), colind->begin(),
                           shape, nnz);

  return view;
}

} // namespace mc
