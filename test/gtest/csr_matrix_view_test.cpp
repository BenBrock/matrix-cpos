#include <gtest/gtest.h>
#include <fmt/core.h>

#include <mc/mc.hpp>

TEST(CsrMatrixView, RowViews) {
	std::size_t m = 10;
	std::size_t n = 300;
  std::size_t nnz = 100;

  using T = int;
  using I = int;

  auto [values, rowptr, colind, shape, _] = mc::generate_csr(m, n, nnz);

  mc::csr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(), shape, nnz);

  for (std::size_t i = 0; i < view.shape()[0]; i++) {
  	auto row = view.row(i);
  	EXPECT_EQ(row.size(), rowptr[i+1] - rowptr[i]);
  	std::size_t j_ptr = rowptr[i];
  	for (auto&& [j, v] : row) {
  		EXPECT_EQ(v, values[j_ptr]);
  		EXPECT_EQ(j, colind[j_ptr]);
  		++j_ptr;
  	}
  }
}

