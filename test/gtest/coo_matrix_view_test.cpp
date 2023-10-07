#include <gtest/gtest.h>

#include <mc/mc.hpp>

TEST(CooMatrixView, RowViews) {
  using T = int;
  using I = int;

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    auto [values, rowind, colind, shape, _] = mc::generate_coo(m, n, nnz);

    mc::coo_matrix_view view(values.begin(), rowind.begin(), colind.begin(),
                             shape, nnz);

    I global_idx = 0;

    for (std::size_t i = 0; i < view.shape()[0]; i++) {
      auto row = view.row(i);

      if (row.size() > 0 && global_idx > 0) {
        EXPECT_TRUE(rowind[global_idx - 1] != rowind[global_idx]);
      }

      for (auto&& [j, v] : row) {
        EXPECT_EQ(i, rowind[global_idx]);
        EXPECT_EQ(j, colind[global_idx]);
        EXPECT_EQ(v, values[global_idx]);
        ++global_idx;
      }
    }
    EXPECT_EQ(global_idx, view.size());

    global_idx = 0;

    // for (auto&& [i, row] : mc::rows(view)) {
    for (auto&& [i, row] : view.row_batch()) {

      if (row.size() > 0 && global_idx > 0) {
        EXPECT_TRUE(rowind[global_idx - 1] != rowind[global_idx]);
      }

      for (auto&& [j, v] : row) {
        EXPECT_EQ(i, rowind[global_idx]);
        EXPECT_EQ(j, colind[global_idx]);
        EXPECT_EQ(v, values[global_idx]);
        ++global_idx;
      }
    }
    EXPECT_EQ(global_idx, view.size());
  }
}
