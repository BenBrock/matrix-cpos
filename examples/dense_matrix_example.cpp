#include <mc/mc.hpp>

#include <vector>
#include <numeric>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  using T = int;

  std::size_t m = 10;
  std::size_t n = 10;
  std::vector<T> v(m*n);

  std::iota(v.begin(), v.end(), 0);

  mc::dense_matrix_view view(v.begin(), {m, n});

  fmt::print("{}\n", v);

  for (std::size_t i = 0; i < view.shape()[0]; i++) {
    for (std::size_t j = 0; j < view.shape()[1]; j++) {
      fmt::print("{:2d}", view[{i, j}]);
      if (j != view.shape()[1]-1) {
        fmt::print(", ");
      }
    }
    fmt::print("\n");
  }

  fmt::print("Rows:\n");
  for (std::size_t i = 0; i < view.shape()[0]; i++) {
    auto row = view.row(i);
    fmt::print("{}\n", row);
    auto row_values = std::ranges::views::values(row);
    fmt::print("{}\n", row_values);
  }

  fmt::print("Columns:\n");
  for (std::size_t j = 0; j < view.shape()[1]; j++) {
    auto col = view.column(j);
    fmt::print("{}\n", col);
    auto col_values = std::ranges::views::values(col);
    fmt::print("{}\n", col_values);
  }

  fmt::print("Diagonals:\n");
  for (std::size_t d = 0; d < view.num_diagonals(); d++) {
    auto diagonal = view.diagonal(d);
    fmt::print("{}\n", diagonal);
    auto diagonal_values = std::ranges::views::values(diagonal);
    fmt::print("{}\n", diagonal_values);
  }

  return 0;
}
