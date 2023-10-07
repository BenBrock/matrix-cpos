#include <fmt/core.h>
#include <fmt/ranges.h>
#include <mc/mc.hpp>

int main(int argc, char** argv) {
  auto&& [values, rowind, colind, shape_, nnz_] =
      mc::generate_coo<float, int>(5, 5, 10);

  mc::coo_matrix_view view(values.begin(), rowind.begin(), colind.begin(),
                           shape_, nnz_);

  for (std::size_t i = 0; i < view.shape()[0]; i++) {
    auto r = view.row(i);
    fmt::print("Row {}: {}\n", i, view.row(i));
  }

  for (auto&& [idx, v] : view) {
    auto&& [i, j] = idx;
    fmt::print("{}, {}: {}\n", i, j, v);
  }

  for (auto&& [i, row] : view.rows()) {
    fmt::print("{}: {}\n", i, row);
  }

  return 0;
}
