#include <mc/mc.hpp>

#include <iostream>
#include <format>

int main(){
  using T = int;
  using I = int;

  for (auto&& [m, n, nnz] :
       {std::tuple(6, 6, 5)}) {
    int block_height = 2;
    int block_width = 2;
    auto [values, rowptr, colind, shape, _] = mc::generate_bcsr(m, n, block_height, block_width, nnz);

    mc::bcsr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(),
                             shape, block_height, block_width, nnz);
    for_each(rowptr.begin(), rowptr.end(), [](int i){
      std::cout << i << " ";
    });
    std::cout << std::endl;
    for_each(colind.begin(), colind.end(), [](int i){
      std::cout << i << " ";
    });
    std::cout << std::endl;
    for_each(values.begin(), values.end(), [](int i){
      std::cout << i << " ";
    });
    std::cout << std::endl;

    auto A = view.blocks();
    for (auto&& [a,b] : A) {

    }


  }

  return 0;
}
