
#pragma once

#include <iterator>

#include <mc/index.hpp>
#include <mc/ranges.hpp>

namespace mc {

template <typename T, std::forward_iterator Iter = T*>
class dense_matrix_view
    : public __ranges::view_interface<dense_matrix_view<T, Iter>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<Iter>;

  using key_type = mc::index<>;
  using map_type = T;

  dense_matrix_view(Iter data, key_type shape)
      : data_(data), shape_(shape), ld_(shape[1]) {}

  dense_matrix_view(Iter data, key_type shape, size_type ld)
      : data_(data), shape_(shape), ld_(ld) {}

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return shape()[0] * shape()[1]; }

  size_type num_diagonals() const noexcept {
    return shape()[0] + shape()[1] - 1;
  }

  scalar_reference operator[](key_type idx) const
    requires(std::random_access_iterator<Iter>)
  {
    return data_[idx[0] * ld_ + idx[1]];
  }

  auto row(size_type row_index) const {
    Iter data = __ranges::next(data_, row_index * ld_);
    __ranges::subrange row_values(data, __ranges::next(data, shape()[1]));

    auto column_indices =
        __ranges::views::iota(size_type(0), size_type(shape()[1]));

    return __ranges::views::zip(column_indices, row_values);
  }

  auto column(size_type column_index) const {
    auto row_indices =
        __ranges::views::iota(size_type(0), size_type(shape()[0]));

    Iter data = __ranges::next(data_, column_index);
    __ranges::subrange data_view(data,
                                 __ranges::next(data_, shape()[0] * ld()));
    auto column_values = __ranges::views::stride(data_view, ld());

    return __ranges::views::zip(row_indices, column_values);
  }

  // NOTE: the diagonal() method allows both negative and positive
  //       diagonal indices. For example, in the matrix:
  //
  //       0 1 2
  //       4 5 6
  //       7 8 9
  //
  // We have the following diagonals
  // - diagonal(0) returns [0, 5, 9]
  // - diagonal(1) returns [1, 6]
  // - diagonal(2) OR diagonal(-2) returns [7]
  // - diagonal(3) OR diagonal(-1) returns [4, 8]
  //
  // This negative indexing is commonly used in many scientific codes.
  // However, we also support positive indexing to support easily
  // iterating through all diagonals.  The negative indexing works
  // similarly to negative array indexing in Python.
  // (x[-1] returns the last element in positive indexing, x[-2] the
  //  next to last element, and so forth.)
  //
  // The two styles can easily be converted back and forth:
  // (Modulos not necessary if we have a range precondition.)
  // negative to positive array index:
  // (d + n) % n
  // Positive to negative array index:
  // (IF greater than highest desired positive index)
  // (d - n) % n

  // Precondition: diagonal_index is in the range [0, num_diagonals())
  auto diagonal(difference_type diagonal_index) const {
    if (diagonal_index < 0) {
      // If given negative diagonal index, convert to positive index.
      // d = (d + n) % n
      // Elide mod because of range precondition.
      diagonal_index = num_diagonals() + diagonal_index;
    }

    if (diagonal_index < shape()[1]) {
      auto diagonal_indices = __ranges::views::iota(
          size_type(0),
          std::min(shape()[0], shape()[1] - size_type(diagonal_index)));

      size_type column_index = diagonal_index;
      Iter data = __ranges::next(data_, diagonal_index);
      __ranges::subrange data_view(data,
                                   __ranges::next(data_, shape()[0] * ld()));

      auto diagonal_values = __ranges::views::stride(data_view, ld() + 1);

      return __ranges::views::zip(diagonal_indices, diagonal_values);
    } else {
      // Convert positive to negative diagonal index.
      // d = (d - n) % n
      // Elide mod because of range precondition.
      difference_type negative_d = diagonal_index - num_diagonals();

      auto diagonal_indices = __ranges::views::iota(
          size_type(0), std::min(shape()[1], shape()[0] + negative_d));

      size_type row_index = -negative_d;
      Iter data = __ranges::next(data_, row_index * ld_);

      __ranges::subrange data_view(data,
                                   __ranges::next(data_, shape()[0] * ld()));

      auto diagonal_values = __ranges::views::stride(data_view, ld() + 1);

      return __ranges::views::zip(diagonal_indices, diagonal_values);
    }
  }

  Iter data() const { return data_; }

  size_type ld() const { return ld_; }

private:
  Iter data_;
  key_type shape_;
  size_type ld_;
};

template <std::random_access_iterator Iter>
dense_matrix_view(Iter, mc::index<>)
    -> dense_matrix_view<std::iter_value_t<Iter>, Iter>;

} // namespace mc
