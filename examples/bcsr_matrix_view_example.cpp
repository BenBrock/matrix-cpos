#include <mc/mc.hpp>

#include <format>
#include <iostream>

#include <fmt/ranges.h>

int main() {
  using T = int;
  using I = int;

  for (auto&& [m, n, nnz] : {std::tuple(6, 6, 5)}) {
    int block_height = 2;
    int block_width = 2;
    auto [values, rowptr, colind, shape, a_nnz] =
        mc::generate_bcsr(m, n, block_height, block_width, nnz);

    mc::bcsr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(),
                              shape, block_height, block_width, nnz);

    fmt::print("values: {}\n", values);
    fmt::print("rowptr: {}\n", rowptr);
    fmt::print("colind: {}\n", colind);

    for (int i_block = 0; i_block < m / block_height; i_block++) {
      for (int j_block_ptr = rowptr[i_block]; j_block_ptr < rowptr[i_block + 1];
           j_block_ptr++) {
        int i_offset = i_block * block_height;
        int j_offset = colind[j_block_ptr];
        int values_offset = j_block_ptr * block_height * block_width;
        auto v = std::ranges::subrange(values.begin() + values_offset,
                                       values.begin() + values_offset +
                                           block_height * block_width);
        fmt::print("A {} x {} block at {}, {} containing values {}\n",
                   block_height, block_width, i_offset, j_offset, v);
      }
    }

    int k = 4;
    auto [b_values, b_shape] = mc::generate_dense(n, k, 123);

    auto [c_values, c_shape] = mc::generate_dense(m, k, 456);

    // std::for_each(b_values.cbegin(), b_values.cend(), [](I v){
    //   std::cout << v << " ";
    // });
    // std::cout << std::endl;
    // std::for_each(c_values.cbegin(), c_values.cend(), [](I v){
    //   std::cout << v << " ";
    // });
    // std::cout << std::endl;
    
    for (auto&& [i, row] : view.blocks()) {
      for (auto&& [j, block] : row) {
        auto block_row_base = i * view.bh();
        auto block_col_base = j;
        for (auto i_ : __ranges::views::iota(I(0), I(view.bh()))) {
          for (auto j_ : __ranges::views::iota(I(0), I(view.bw()))) {
            if (0 == block[{i_, j_}]) continue;
            
            for (auto k_ : __ranges::views::iota(I(0), I(k))) {
              auto row_addr = block_row_base + i_;
              auto col_addr = block_col_base + j_;
              I b_addr = col_addr * k + k_;
              I c_addr = row_addr * k + k_;
              // std::cout << c_values[c_addr] << "+=" << block[{i_, j_}] << "*" << b_values[b_addr] << std::endl;
              c_values[c_addr] += block[{i_, j_}] * b_values[b_addr];
            }
          }
        } 
      }
    }
    // std::for_each(c_values.cbegin(), c_values.cend(), [](I v){
    //   std::cout << v << " ";
    // });
    // std::cout << std::endl;
  }

  return 0;
}