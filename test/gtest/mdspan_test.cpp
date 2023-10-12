#include <gtest/gtest.h>

#include <mc/mc.hpp>

TEST(MdspanView, RowColumnViews) {
  using T = int;
  using I = int;

  for (auto&& [m, n, nnz] :
       {std::tuple(1000, 100, 100), std::tuple(100, 1000, 10000),
        std::tuple(40, 40, 1000)}) {
    auto [x, shape] = mc::generate_dense(m, n);
    std::experimental::mdspan a(x.data(), shape[0], shape[1]);

    for (auto&& [i, row] : mc::rows(a)) {
      for (auto&& [j, value] : row) {
        EXPECT_EQ(value, (a[i, j]));
      }
    }

    for (auto&& [j, column] : mc::columns(a)) {
      for (auto&& [i, value] : column) {
        EXPECT_EQ(value, (a[i, j]));
      }
    }
  }
}
