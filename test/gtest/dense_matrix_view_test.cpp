#include <gtest/gtest.h>

#include <mc/mc.hpp>
#include <vector>
#include <numeric>

TEST(DenseMatrixView, RowColumnViews) {
  std::size_t m = 100;
  std::size_t n = 100;

  using T = int;

  std::vector<T> v(m*n);

  std::iota(v.begin(), v.end(), 0);

  mc::dense_matrix_view view(v.begin(), {m, n});

  for (std::size_t i = 0; i < view.shape()[0]; i++) {
    auto row = view.row(i);

    EXPECT_EQ(row.size(), view.shape()[1]);

    std::size_t column_count = 0;
    for (auto&& [j, v] : row) {
      EXPECT_EQ(j, column_count);
      T scalar_v = view[{i, j}];
      EXPECT_EQ(v, scalar_v);

      ++column_count;
    }
  }

  for (std::size_t j = 0; j < view.shape()[1]; j++) {
    auto column = view.column(j);

    EXPECT_EQ(column.size(), view.shape()[0]);

    std::size_t row_count = 0;
    for (auto&& [i, v] : column) {
      EXPECT_EQ(i, row_count);
      T scalar_v = view[{i, j}];
      EXPECT_EQ(v, scalar_v);

      ++row_count;
    }
  }
}

