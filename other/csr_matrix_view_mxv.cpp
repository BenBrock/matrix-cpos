#include <gtest/gtest.h>

#include <mc/mc.hpp>

TEST(CsrMatrixMxv, RowViews) {
  using T = int;
  using I = int;

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    auto [values, rowptr, colind, shape, _] = mc::generate_csr(m, n, nnz);

    auto [vec, length, nnzv] = mc::generate_vec(n, nnz);

    mc::csr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(),
                             shape, nnz);

    std::vector<T> result(n);

    std::iota(result.begin(), result.end(), 0);

    for (std::size_t i = 0; i < view.shape()[0]; i++) {
      auto row = view.row(i);
      EXPECT_EQ(row.size(), rowptr[i + 1] - rowptr[i]);
      for (auto&& [j, v] : row) {
        result[i] += v * vec[j];
      }
    }
  }
}
