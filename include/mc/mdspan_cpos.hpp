#pragma once

#include <experimental/mdspan>
#include <mc/cpos.hpp>

namespace mc {

// Generic implementation of rows customization point
// for any mdspan.
template <typename T, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
auto tag_invoke(
    mc::rows_fn_,
    std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>
        mdspan) {
  using index_type = decltype(mdspan.extent(0));
  using reference = std::experimental::mdspan<T, Extents, LayoutPolicy,
                                              AccessorPolicy>::reference;

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

// Generic implementation of columns customization point
// for any mdspan.
template <typename T, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
auto tag_invoke(
    mc::columns_fn_,
    std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>
        mdspan) {
  using index_type = decltype(mdspan.extent(0));
  using reference = std::experimental::mdspan<T, Extents, LayoutPolicy,
                                              AccessorPolicy>::reference;

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

// Specialization of rows customization point
// for layout right matrices.
template <typename T, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
  requires(
      std::is_same_v<LayoutPolicy, std::experimental::layout_right> &&
      std::is_same_v<AccessorPolicy, std::experimental::default_accessor<T>>)
auto tag_invoke(
    mc::rows_fn_,
    std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>
        mdspan) {
  using index_type = decltype(mdspan.extent(0));
  using reference = std::experimental::mdspan<T, Extents, LayoutPolicy,
                                              AccessorPolicy>::reference;

  auto data = mdspan.data_handle();

  auto row_indices = __ranges::views::iota(index_type(0), mdspan.extent(0));

  auto rows = row_indices | __ranges::views::transform([=](auto row_index) {
                auto column_indices =
                    __ranges::views::iota(index_type(0), mdspan.extent(1));
                auto first = data + row_index * mdspan.extent(1);
                auto values =
                    __ranges::subrange(first, first + mdspan.extent(1));

                return __ranges::views::zip(column_indices, values);
              });

  return __ranges::views::zip(row_indices, rows);
}

} // namespace mc
