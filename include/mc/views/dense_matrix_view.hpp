
#pragma once

#include <iterator>

#include <mc/ranges.hpp>
#include <mc/index.hpp>

namespace mc {

template <typename T, typename Iter = T *>
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

  scalar_reference operator[](key_type idx) const {
    return data_[idx[0] * ld_ + idx[1]];
  }

  auto row(size_type row_index) const {
    Iter data = data_ + row_index * ld_;
    __ranges::subrange row_values(data, data + shape()[1]);

    auto column_indices = __ranges::views::iota(size_type(0), size_type(shape()[1]));

    return __ranges::views::zip(column_indices, row_values);
  }

  auto column(size_type column_index) const {
    auto row_indices =
        __ranges::views::iota(size_type(0), size_type(shape()[0]));

    Iter data = data_ + column_index;
    __ranges::subrange data_view(data, data_ + shape()[0]*ld_);
    size_type ld = ld_;
    auto column_values = __ranges::views::stride(data_view, ld);

    return __ranges::views::zip(row_indices, column_values);
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

} // end mc
