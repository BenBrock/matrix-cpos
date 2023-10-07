#pragma once

#include <iterator>

#include <mc/index.hpp>
#include <mc/ranges.hpp>

namespace mc {

template <typename T, typename I, std::forward_iterator TIter = T*,
          std::forward_iterator IIter = I*>
class dcsr_matrix_view
    : public __ranges::view_interface<dcsr_matrix_view<T, I, TIter, IIter>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<TIter>;

  using scalar_type = T;
  using index_type = I;

  using key_type = mc::index<I>;
  using map_type = T;

  dcsr_matrix_view(TIter values, IIter rowind, IIter rowptr, IIter colind,
                   key_type shape, I rowind_size, size_type nnz)
      : values_(values), rowind_(rowind), rowptr_(rowptr), colind_(colind),
        shape_(shape), rowind_size_(rowind_size), nnz_(nnz) {}

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return nnz_; }

  auto rows() const {
    auto row_indices = __ranges::subrange(rowind_, rowind_ + rowind_size_);

    auto row_values =
        __ranges::views::iota(I(0), I(rowind_size_)) |
        __ranges::views::transform([*this](auto i_ptr) {
          auto column_indices = __ranges::subrange(
              colind_ + rowptr_[i_ptr], colind_ + rowptr_[i_ptr + 1]);
          auto values = __ranges::subrange(values_ + rowptr_[i_ptr],
                                           values_ + rowptr_[i_ptr + 1]);
          return __ranges::views::zip(column_indices, values);
        });

    return __ranges::views::zip(row_indices, row_values);
  }

  auto values_data() const { return values_; }
  auto rowptr_data() const { return rowptr_; }
  auto rowind_data() const { return rowind_; }
  auto colind_data() const { return colind_; }

private:
  TIter values_;
  IIter rowind_;
  IIter rowptr_;
  IIter colind_;

  key_type shape_;
  I rowind_size_;
  size_type nnz_;
};

template <typename TIter, typename IIter, typename... Args>
dcsr_matrix_view(TIter, IIter, IIter, Args&&...)
    -> dcsr_matrix_view<std::iter_value_t<TIter>, std::iter_value_t<IIter>,
                        TIter, IIter>;

} // namespace mc
