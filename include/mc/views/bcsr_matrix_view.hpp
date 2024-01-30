#pragma once

#include <iostream>
#include <iterator>
#include <utility>
#include <vector>
#include <algorithm>

#include <mc/index.hpp>
#include <mc/ranges.hpp>

namespace mc {

template <typename T, typename I, std::forward_iterator TIter = T*,
          std::forward_iterator IIter = I*>
class bcsr_matrix_view
    : public __ranges::view_interface<bcsr_matrix_view<T, I, TIter, IIter>> {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_reference = std::iter_reference_t<TIter>;

  using scalar_type = T;
  using index_type = I;

  using key_type = mc::index<I>;
  using map_type = T;

  bcsr_matrix_view(TIter values, IIter rowptr, IIter colind, key_type shape,
                   size_type block_height, size_type block_width, size_type nnz)
      : values_(values), rowptr_(rowptr), colind_(colind), shape_(shape),
        block_height_(block_height), block_width_(block_width), nnz_(nnz) {}

  key_type shape() const noexcept { return shape_; }

  size_type bh() const noexcept { return block_height_; }
  size_type bw() const noexcept { return block_width_; }

  size_type size() const noexcept { return nnz_; }

  struct Block {
    std::vector<T> block_values;
    I block_height_;
    I block_width_;
    Block(std::ranges::input_range auto&& range, I block_height, I block_width) 
    {
      block_height_ = block_height;
      block_width_ = block_width;
      for (auto&& e : range) {
        block_values.push_back(T(e));
      }
    }

    const Block operator=(const Block& block) {
      for (auto&& e : block.block_values) {
        block_values.push_back(std::move(e));
      }
      block_height_ = block.block_height_;
      block_width_ = block.block_width_;
      return *this;
    }
    const T operator[](std::vector<I> index) {
      auto addr = index[0] * block_width_ + index[1];
      return block_values[addr];
    }

    size_type bw() const noexcept { return block_width_; }

    Block(const Block& block) {
      for (auto&& e : block.block_values) {
        block_values.push_back(std::move(e));
      }
      block_height_ = block.block_height_;
      block_width_ = block.block_width_;
    }

    void print() const {
      for (auto&& e : block_values) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
    }
  };

  auto row_blocks(I row_index) const {
    I first = rowptr_[row_index];
    I last = rowptr_[row_index + 1];

    __ranges::subrange column_indices(__ranges::next(colind_data(), first),
                                      __ranges::next(colind_data(), last));
    
    __ranges::subrange index_range = __ranges::views::iota(I(first), I(last));

    auto row_block_values = 
      index_range | __ranges::views::transform(
      [*this](auto index) {
        I start = index*bh()*bw();
        I end = (index+1)*bh()*bw();
        __ranges::subrange block_values(__ranges::next(values_data(), start),
                                        __ranges::next(values_data(), end));
        Block A(block_values, bh(), bw());
        return A;
    });
    return __ranges::views::zip(column_indices, row_block_values);
  }

  auto blocks() const {
    I block_end = shape()[0] / bh();
    auto row_indices = __ranges::views::iota(I(0), I(block_end));

    auto row_values = 
      row_indices | __ranges::views::transform(
        [*this](auto row_index){ return row_blocks(row_index); }
      );
    return std::ranges::views::zip(row_indices, row_values);
  }

  auto values_data() const { return values_; }
  auto rowptr_data() const { return rowptr_; }
  auto colind_data() const { return colind_; }

private:
  TIter values_;
  IIter rowptr_;
  IIter colind_;

  key_type shape_;
  size_type block_height_;
  size_type block_width_;
  size_type nnz_;
};

template <typename TIter, typename IIter, typename... Args>
bcsr_matrix_view(TIter, IIter, IIter, Args&&...)
    -> bcsr_matrix_view<std::iter_value_t<TIter>, std::iter_value_t<IIter>,
                        TIter, IIter>;

} // namespace mc
