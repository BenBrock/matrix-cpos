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

    
  }

  return 0;
}
