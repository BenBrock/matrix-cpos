#pragma once

#include <mc/index.hpp>
#include <iterator>

namespace mc {

template <typename T, typename I, std::forward_iterator TIter = T *, std::forward_iterator IIter = I *>
class csr_matrix_view
    : public __ranges::view_interface<csr_matrix_view<T, I, TIter, IIter>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<TIter>;

  using scalar_type = T;
  using index_type = I;

  using key_type = mc::index<I>;
  using map_type = T;

  csr_matrix_view(TIter values, IIter rowptr, IIter colind, key_type shape,
                  size_type nnz)
      : values_(values), rowptr_(rowptr), colind_(colind), shape_(shape),
        nnz_(nnz) {}

  key_type shape() const noexcept { return shape_; }

  size_type size() const noexcept { return nnz_; }

  auto row(I row_index) const {
    I first = rowptr_[row_index];
    I last = rowptr_[row_index + 1];

    __ranges::subrange row_values(__ranges::next(values_data(), first),
                                  __ranges::next(values_data(), last));
    __ranges::subrange column_indices(__ranges::next(colind_data(), first),
                                      __ranges::next(colind_data(), last));

    return __ranges::views::zip(column_indices, row_values);
  }

  auto values_data() const { return values_; }

  auto rowptr_data() const { return rowptr_; }

  auto colind_data() const { return colind_; }

private:
  TIter values_;
  IIter rowptr_;
  IIter colind_;

  key_type shape_;
  size_type nnz_;
};

template <typename TIter, typename IIter, typename... Args>
csr_matrix_view(TIter, IIter, IIter, Args &&...)
    -> csr_matrix_view<std::iter_value_t<TIter>, std::iter_value_t<IIter>,
                       TIter, IIter>;

} // end mc