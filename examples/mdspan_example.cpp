#include <mc/mc.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  std::size_t m = 10;
  std::size_t n = 10;

  using T = int;
  using I = int;

  auto [x, shape] = mc::generate_dense(m, n);

  std::experimental::mdspan a(x.data(), shape[0], shape[1]);

  for (std::size_t i = 0; i < a.extent(0); i++) {
    for (std::size_t j = 0; j < a.extent(1); j++) {
      a[i, j] = 12;
    }
  }

  for (auto&& [i, row] : mc::rows(a)) {
    for (auto&& [j, value] : row) {
      value = i;
    }
  }

  for (auto&& [i, row] : mc::rows(a)) {
    auto values = std::ranges::views::values(row);
    fmt::print("{}\n", values);
  }

  for (auto&& [i, row] : mc::columns(a)) {
    auto values = std::ranges::views::values(row);
    fmt::print("{}\n", values);
  }

  return 0;
}
