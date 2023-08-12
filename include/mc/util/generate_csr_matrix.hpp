#pragma once

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

namespace mc {

template <typename T = float, typename I = int>
auto generate_csr(I m, I n, std::size_t nnz, std::size_t seed = 0) {
  std::vector<T> values;
  std::vector<I> colind;

  values.reserve(nnz);
  colind.reserve(nnz);

  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, n-1);
  std::uniform_real_distribution d_f(0.0, 100.0);
  std::uniform_int_distribution<I> d_m(0, nnz);

  for (std::size_t i = 0; i < nnz; i++) {
  	colind.push_back(d(g));
  }

  for (std::size_t i = 0; i < nnz; i++) {
  	values.push_back(d_f(g));
  }

  std::vector<I> rowptr;
  rowptr.reserve(m+1);
  rowptr.push_back(0);
  for (std::size_t i = 0; i < m-1; i++) {
  	rowptr.push_back(d_m(g));
  }
  rowptr.push_back(nnz);

  std::sort(rowptr.begin(), rowptr.end());

  for (std::size_t i = m; i >= 1; i--) {
  	rowptr[i] -= rowptr[i-1];
  }

  std::inclusive_scan(rowptr.begin(), rowptr.end(), rowptr.begin());

  return std::tuple(values, rowptr, colind, mc::index<I>(m, n), I(nnz));
}

} // end mc