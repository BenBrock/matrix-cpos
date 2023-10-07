#include <mc/mc.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <experimental/mdspan>

template <typename... Args>
auto rows(std::experimental::mdspan<Args...> mdspan) {
  using index_type = decltype(mdspan.extent(0));
  using reference = std::experimental::mdspan<Args...>::reference;

  auto row_indices = __ranges::views::iota(index_type(0), mdspan.extent(0));

  auto rows = row_indices | __ranges::views::transform([=](auto row_index) {
                auto column_indices =
                    __ranges::views::iota(index_type(0), mdspan.extent(1));
                auto values = column_indices |
                              __ranges::views::transform(
                                  [=](auto column_index) -> reference {
                                    return mdspan[row_index, column_index];
                                  });
                return __ranges::views::zip(column_indices, values);
              });
  return __ranges::views::zip(row_indices, rows);
}

template <typename... Args>
auto columns(std::experimental::mdspan<Args...> mdspan) {
  using index_type = decltype(mdspan.extent(0));
  using reference = std::experimental::mdspan<Args...>::reference;

  auto column_indices = __ranges::views::iota(index_type(0), mdspan.extent(1));

  auto columns =
      column_indices | __ranges::views::transform([=](auto column_index) {
        auto row_indices =
            __ranges::views::iota(index_type(0), mdspan.extent(0));
        auto values =
            row_indices |
            __ranges::views::transform([=](auto row_index) -> reference {
              return mdspan[row_index, column_index];
            });
        return __ranges::views::zip(row_indices, values);
      });
  return __ranges::views::zip(column_indices, columns);
}

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

  for (auto&& [i, row] : rows(a)) {
    for (auto&& [j, value] : row) {
      value = i;
    }
  }

  for (auto&& [i, row] : rows(a)) {
    auto values = std::ranges::views::values(row);
    fmt::print("{}\n", values);
  }

  for (auto&& [i, row] : columns(a)) {
    auto values = std::ranges::views::values(row);
    fmt::print("{}\n", values);
  }

  return 0;
}
