#include <chrono>
#include <mc/mc.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  std::size_t m = 10;
  std::size_t n = 300;
  std::size_t nnz = 100;

  using T = int;
  using I = int;

  auto [values, rowptr, colind, shape, _] = mc::generate_csr<T, I>(m, n, nnz);

  mc::csr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(),
                           shape, nnz);

  for (std::size_t i = 0; i < view.shape()[0]; i++) {
    auto row = view.row(i);
    fmt::print("Row {}: {}\n", i, row);
  }

  fmt::print("Printing out rows:\n");
  for (auto&& [row_index, row] : view.rows()) {
    fmt::print("Row {}: {}\n", row_index, row);
  }

  auto a = view;

  std::vector<T> c(m, 0);
  std::vector<T> b(n, 1);

  return 0;
}
