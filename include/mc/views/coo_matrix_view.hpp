#pragma once

#include <algorithm>
#include <execution>
#include <iterator>
#include <numeric>
#include <vector>

#include <mc/index.hpp>
#include <mc/ranges.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace mc {

template <typename T, typename I, std::forward_iterator TIter = T*,
          std::forward_iterator IIter = I*>

class coo_matrix_view
    : public __ranges::view_interface<coo_matrix_view<T, I, TIter, IIter>> {
public:
  using size_type = std::size_t;
  using difference_type = I;

  using scalar_reference = std::iter_reference_t<TIter>;

  using scalar_type = T;
  using index_type = I;

  using key_type = mc::index<I>;
  using map_type = T;

  coo_matrix_view(TIter values, IIter rowind, IIter colind, key_type shape,
                  size_type nnz)
      : values_(values), rowind_(rowind), colind_(colind), shape_(shape),
        nnz_(nnz),
        values_view_(__ranges::views::zip(
            __ranges::views::zip(
                __ranges::subrange(rowind_, __ranges::next(rowind_, nnz_)),
                __ranges::subrange(colind_, __ranges::next(colind_, nnz_))),
            __ranges::subrange(values_, __ranges::next(values_, nnz_)))) {}

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return nnz_; }

  auto values_data() const { return values_; }
  auto rowind_data() const { return rowind_; }
  auto colind_data() const { return colind_; }

  auto begin() const { return values_view_.begin(); }

  auto end() const { return values_view_.end(); }

  auto rows() const {
    auto row_indices = __ranges::views::iota(I(0), I(shape()[0]));

    auto rows =
        row_indices | __ranges::views::transform(
                          [*this](auto row_index) { return row(row_index); });

    return __ranges::views::zip(row_indices, rows);
  }

  auto row(I row_index) const {
    auto [first_it, last_it] =
        std::equal_range(rowind_, rowind_ + nnz_, row_index);

    auto first = first_it - rowind_;
    auto last = last_it - rowind_;

    auto values = __ranges::subrange(values_ + first, values_ + last);
    auto colind = __ranges::subrange(colind_ + first, colind_ + last);

    return __ranges::views::zip(colind, values);
  }

private:
  auto row_batch() const {
    auto rowind = __ranges::subrange(rowind_, rowind_ + nnz_);

    auto view =
        rowind | __ranges::views::enumerate | __ranges::views::adjacent<2>;

    std::vector<I> rowptr(shape()[0] + 1, I(0));

    std::for_each(std::execution::par_unseq, view.begin(), view.end(),
                  [&](auto&& x) {
                    auto&& [i0, x0] = std::get<0>(x);
                    auto&& [i1, x1] = std::get<1>(x);
                    if (x1 - x0 > 0) {
                      rowptr[x0 + 1] = i1;
                    }
                  });

    rowptr[rowind.back() + 1] = nnz_;

    std::inclusive_scan(std::execution::par_unseq, rowptr.begin(), rowptr.end(),
                        rowptr.begin(),
                        [](I a, I b) { return (a < b) ? b : a; });

    return std::move(rowptr) | __ranges::views::enumerate |
           __ranges::views::adjacent<2> |
           __ranges::views::transform([*this](auto&& x) {
             auto&& [row_index, first] = std::get<0>(x);
             auto&& [_, last] = std::get<1>(x);

             auto values = __ranges::subrange(values_ + first, values_ + last);
             auto column_indices =
                 __ranges::subrange(colind_ + first, colind_ + last);

             auto row = __ranges::views::zip(column_indices, values);
             return std::pair(row_index, row);
           });
  }

  using __values_range_type = decltype(__ranges::subrange(
      std::declval<TIter>(), std::declval<TIter>()));
  using __rowind_range_type = decltype(__ranges::subrange(
      std::declval<IIter>(), std::declval<IIter>()));
  using __colind_range_type = __rowind_range_type;

  using __indices_zip_type =
      decltype(__ranges::views::zip(std::declval<__rowind_range_type>(),
                                    std::declval<__colind_range_type>()));
  using __values_zip_type = decltype(__ranges::views::zip(
      std::declval<__indices_zip_type>(), std::declval<__values_range_type>()));

  TIter values_;
  IIter rowind_;
  IIter colind_;

  key_type shape_;
  size_type nnz_;

  __values_zip_type values_view_;
};

template <typename TIter, typename IIter, typename... Args>
coo_matrix_view(TIter, IIter, IIter, Args&&...)
    -> coo_matrix_view<std::iter_value_t<TIter>, std::iter_value_t<IIter>,
                       TIter, IIter>;

} // namespace mc
