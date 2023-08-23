#include <gtest/gtest.h>

#include <mc/mc.hpp>
#include <numeric>
#include <vector>

TEST(DenseMatrixView, RowColumnViews) {
  using T = int;

  for (auto&& [m, n] :
       {mc::index<>(1000, 100), mc::index<>(100, 1000), mc::index<>(40, 40)}) {
    std::vector<T> v(m * n);

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
}

TEST(DenseMatrixView, DiagonalViews) {
  using T = int;

  for (auto&& [m, n] :
       {mc::index<>(1000, 100), mc::index<>(100, 1000), mc::index<>(40, 40)}) {
    std::vector<T> v(m * n);

    std::iota(v.begin(), v.end(), 0);

    mc::dense_matrix_view view(v.begin(), {m, n});

    for (std::ptrdiff_t d = 0; d < view.shape()[1]; d++) {
      // fmt::print("diagonal: {}\n", view.diagonal(d));
      auto diagonal = view.diagonal(d);
      std::size_t diagonal_size =
          std::min(view.shape()[0], view.shape()[1] - d);
      EXPECT_EQ(diagonal.size(), diagonal_size);
      assert(diagonal.size() == diagonal_size);

      for (std::size_t e = 0; e < diagonal_size; e++) {
        std::size_t i = e;
        std::size_t j = e + d;
        auto diagonal_value = view[{i, j}];
        auto&& [d, diagonal_value_] = diagonal[e];
        EXPECT_EQ(diagonal_value_, diagonal_value);
        EXPECT_EQ(d, e);
      }
    }

    for (std::ptrdiff_t d = -1; d > -std::ptrdiff_t(view.shape()[0]); d--) {
      auto diagonal = view.diagonal(d);
      std::size_t diagonal_size =
          std::min(view.shape()[1], view.shape()[0] + d);
      EXPECT_EQ(diagonal.size(), diagonal_size);
      for (std::size_t e = 0; e < diagonal_size; e++) {
        std::size_t i = e - d;
        std::size_t j = e;
        auto diagonal_value = view[{i, j}];
        auto&& [d, diagonal_value_] = diagonal[e];
        EXPECT_EQ(diagonal_value_, diagonal_value);
        EXPECT_EQ(d, e);
      }
    }

    for (std::ptrdiff_t d = 0; d < view.num_diagonals(); d++) {
      auto diagonal = view.diagonal(d);

      std::ptrdiff_t d_neg = d;

      if (d >= view.shape()[1]) {
        d_neg -= std::ptrdiff_t(view.num_diagonals());
      }

      std::size_t diagonal_size;
      if (d_neg >= 0) {
        diagonal_size = std::min(view.shape()[0], view.shape()[1] - d_neg);
      } else {
        diagonal_size = std::min(view.shape()[1], view.shape()[0] + d_neg);
      }
      EXPECT_EQ(diagonal.size(), diagonal_size);

      for (std::size_t e = 0; e < diagonal_size; e++) {
        std::size_t i, j;
        if (d_neg >= 0) {
          i = e;
          j = e + d_neg;
        } else {
          i = e - d_neg;
          j = e;
        }
        auto diagonal_value = view[{i, j}];
        auto&& [d, diagonal_value_] = diagonal[e];
        EXPECT_EQ(diagonal_value_, diagonal_value);
        EXPECT_EQ(d, e);
      }
    }
  }
}
