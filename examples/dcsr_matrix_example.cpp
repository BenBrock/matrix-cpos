#include <mc/mc.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  std::size_t m = 100;
  std::size_t n = 300;
  std::size_t nnz = 100;

  using T = int;
  using I = int;

  auto [values, rowind, rowptr, colind, shape, rowind_size, _] =
      mc::generate_dcsr<T, I>(m, n, nnz);

  mc::dcsr_matrix_view view(values.begin(), rowind.begin(), rowptr.begin(),
                            colind.begin(), shape, rowind_size, nnz);

  fmt::print("Printing out rows:\n");
  for (auto&& [row_index, row] : view.rows()) {
    fmt::print("Row {}: {}\n", row_index, row);
  }

  return 0;
}
