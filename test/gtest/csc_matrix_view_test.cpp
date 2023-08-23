#include <gtest/gtest.h>

#include <mc/mc.hpp>

TEST(CscMatrixView, RowViews) {
  using T = int;
  using I = int;

  for (auto&& [m, n, nnz] : {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000), std::tuple(40, 40, 1000)}) {
    auto [values, colptr, rowind, shape, _] = mc::generate_csc(m, n, nnz);

    mc::csc_matrix_view view(values.begin(), colptr.begin(), rowind.begin(), shape, nnz);

    for (std::size_t j = 0; j < view.shape()[1]; j++) {
      auto column = view.column(j);
      EXPECT_EQ(column.size(), colptr[j+1] - colptr[j]);
      std::size_t i_ptr = colptr[j];
      for (auto&& [i, v] : column) {
        EXPECT_EQ(v, values[i_ptr]);
        EXPECT_EQ(i, rowind[i_ptr]);
        ++i_ptr;
      }
    }
  }
}

