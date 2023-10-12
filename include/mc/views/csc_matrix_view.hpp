#pragma once

#include <iterator>

#include <mc/index.hpp>
#include <mc/ranges.hpp>
#include <mc/views/csr_matrix_view.hpp>

namespace mc {

template <typename T, typename I, std::forward_iterator TIter = T*,
          std::forward_iterator IIter = I*>
class csc_matrix_view
    : public __ranges::view_interface<csc_matrix_view<T, I, TIter, IIter>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<TIter>;

  using scalar_type = T;
  using index_type = I;

  using key_type = mc::index<I>;
  using map_type = T;

  csc_matrix_view(TIter values, IIter colptr, IIter rowind, key_type shape,
                  size_type nnz)
      : matrix_t_(values, colptr, rowind, {shape[1], shape[0]}, nnz) {}

  key_type shape() const noexcept {
    return {matrix_t_.shape()[1], matrix_t_.shape()[0]};
  }

  size_type size() const noexcept { return matrix_t_.size(); }

  auto column(I column_index) const { return matrix_t_.row(column_index); }

  auto columns() const { return matrix_t_.rows(); }

  auto values_data() const { return matrix_t_.values_data(); }

  auto colptr_data() const { return matrix_t_.rowptr_data(); }

  auto rowind_data() const { return matrix_t_.colind_data(); }

private:
  csr_matrix_view<T, I, TIter, IIter> matrix_t_;
};

template <typename TIter, typename IIter, typename... Args>
csc_matrix_view(TIter, IIter, IIter, Args&&...)
    -> csc_matrix_view<std::iter_value_t<TIter>, std::iter_value_t<IIter>,
                       TIter, IIter>;

} // namespace mc
