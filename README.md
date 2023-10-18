# Tools for Multi-Dimensional Iteration over Sparse Matrices

This repo contains a collection of tools for high-level, multi-dimensional
iteration over sparse matrices in C++.  It uses the ranges library to implement
a number of sparse matrix views, customization points, and concepts for sparse
matrix iteration, along with algorithms that use those iteration concepts.

## Example

The following example demonstrates the `csr_matrix_view` view.  The
`csr_matrix_view` is a high-level view of a traditional CSR sparse matrix data
structure.  A CSR supports the `row_iterable` concept, meaning we can iterate
over it by row.  Other matrices can support the `row_iterable` concept as well,
such as a dense matrix, doubly compressed sparse row (DCSR) matrix, or block
sparse format matrix.  Other iteration concepts include `column_iterable` and
`diagonal_iterable`, and can be expanded to include more advanced types of
iteration such as iteration over tiles of a matrix or iteration with a
particular work distribution of work stealing strategy.

```cpp
#include <mc/mc.hpp>

// The row_iterable concept ensures that we can iterate
// over rows.
template <mc::row_iterable A,
          std::ranges::random_access_range C,
          std::ranges::random_access_range B>
void spmv(C&& c, A&& a, B&& b) {
  for (auto&& [i, row] : mc::rows(a)) {
    for (auto&& [j, a_v] : row) {
      c[i] += a_v * b[j];
    }
  }
}


int main(int argc, char** argv) {
  using T = float;
  using I = int;
  int m, n, nnz;
  m = n = 1000000;
  nnz = 1000;

  // Generate a sparse matrix
  auto [values, rowptr, colind, shape, _] = mc::generate_csr<T, I>(m, n, nnz);

  // Create a sparse matrix view
  mc::csr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(),
                           shape, nnz);

  std::vector<T> c(m, 0);
  std::vector<T> b(n, 1);

  // Execute algorithm
  spmv(c, a, b);

  return 0;
}
```
