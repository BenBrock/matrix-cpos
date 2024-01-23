#pragma once

#include <iterator>
#include <utility>
#include <iostream>

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
        block_height_(block_height), block_width_(block_width), nnz_(nnz) {
      }

  key_type shape() const noexcept { return shape_; }

  size_type bh() const noexcept { return block_height_; }
  size_type bw() const noexcept { return block_width_; }

  size_type size() const noexcept { return nnz_; }

  struct Block {
    std::vector<T> block_values;
    Block(std::ranges::input_range auto&& range) {
      for (auto&& e : range) {
        block_values.push_back(T(e));
      }
    }
    
    const Block operator=(const Block& block) {
      for (auto && e : block.block_values) {
        block_values.push_back(std::move(e));
      }
    }
    Block(const Block& block){
      for (auto && e : block.block_values) {
        block_values.push_back(std::move(e));
      }
    }
    
    void print() const {
      for (auto && e : block_values) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
    }
  };

  auto block(std::pair<I, I> block_index) const {
    I row_index = block_index.first;
    I index = block_index.second;
    I column_index = colind_[index];

    I first = index*bh()*bw();
    I last = (index+1)*bh()*bw();

    __ranges::subrange block_values(__ranges::next(values_data(), first),
                                    __ranges::next(values_data(), last));

    Block block_unit(block_values);

    return block_unit;
  }

  auto blocks() const {
    I block_end = shape()[0] / bh();
    auto row_indices = __ranges::views::iota(I(0), I(block_end));
    
    std::vector<std::pair<I, I>> row_and_id_indices;
    for (auto row_index : row_indices) {
      I first = rowptr_[row_index];
      I last = rowptr_[row_index + 1];
      for (auto index : __ranges::views::iota(I(first), I(last))) {
        row_and_id_indices.push_back(std::make_pair(row_index, index));
      }
    }
    // for_each(block_indices.begin(), block_indices.end(), [](std::pair<I, I> p){
    //   std::cout << "(" << p.first << ", " << p.second << ")";
    // });
    // std::cout << std::endl;
    auto block_indices = 
        row_and_id_indices | __ranges::views::transform(
          [*this](std::pair<I,I> row_and_id_index) { return std::make_pair(row_and_id_index.first*bh(), colind_[row_and_id_index.second]); });

    auto block_values =
        row_and_id_indices | __ranges::views::transform(
                          [*this](auto block_index) { return block(block_index); });

    // for (auto&& index : block_indices) {
    //   std::cout << "(" << index.first << "," << index.second << ")";
    // }

    // for (auto&& block : block_values) {
    //   block.print();
    // }

    return std::ranges::views::zip(row_indices, block_values);
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
