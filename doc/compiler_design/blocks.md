# `block` Customization Point Design

<!-- vscode-markdown-toc -->
* [Motivation](#Motivation)
* [Construction](#Construction)
	* [BCSR Construction](#BCSRConstruction)
	* [Using MdSpan](#UsingMdSpan)
* [Application](#Application)
	* [SpMV](#SpMV)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Motivation'></a>Motivation

Existing customization points include `row`, `column` and `diagonal`. Besides these, block view is known as a common view for parallelism. Therefore, `block` view is adopted as a new cp.

## <a name='Construction'></a>Construction
Here we implement BCSR format as a representative blocked sparse format. 

### <a name='BCSRConstruction'></a>BCSR Construction

```
mc::bcsr_matrix_view view(values.begin(), rowptr.begin(), colind.begin(),
                          shape, block_height, block_width, nnz);
```

We need to pass 7 arguments to construct a BCSR view.
+ `values` is the array contains the entries of blocks from original matrix;
+ `rowptr` is the array contains the starting point of each row with block view in `values` array;
+ `colind` is the array contains the column index of each block;
+ `shape` is the size of the original matrix;
+ `block_height` is the first dimension of block;
+ `block_width` is the second dimension of block;
+ `nnz` is the number of non-zero elements in original matrix.

For example, for matrix $A$ as follow:
$$
A = \left(
\begin{matrix}
0 & 2.42  & 0 & 0 & 0 & 0 \\
59.26 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
85.34 & 91.42 & 82.82 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
\end{matrix}
\right)
$$
Its `values`, `rowptr` and `colind` arrays are as follows:
```
values: [0, 2.42, 59.26, 0, 0, 0, 85.34, 91.42, 0, 0, 82.82, 0]
rowptr: [0, 1, 3, 3]
colind: [0, 0, 2]
```
It's recommended to use provided `mc::generate_bcsr` function to directly generate random benchmark by providing the size of matrix and block and number of non-zeros.
```
auto [values, rowptr, colind, shape, a_nnz] =
    mc::generate_bcsr(m, n, block_height, block_width, nnz);
```

### <a name='UsingMdSpan'></a>Using MdSpan

Another idea is to use `std::mdspan` to construct views for BCSR format.

```c++
auto [values, rowptr, colind, shape, a_nnz] =
    mc::generate_bcsr(m, n, block_height, block_width, nnz);

std::experimental::mdspan a(values(), m, n);

for (auto && [{bx, by}, block] : mc::blocks(a)) {
  auto values = std::ranges::views::values(block);
  fmt::print("A {} x {} block at {}, {} containing values {}\n",
                   block_height, block_width, bx, by, values);
}
```

## <a name='Application'></a>Application 

### <a name='SpMV'></a>SpMV

The processing flow of SpMV $c=Ab$ is designed as follows:
+ Iterate over each block in a sparse matrix. The block iterator is provided by specific interface. The details is transparent to user.
+ Iterate over each element in block and calculate its indices to determine the corresponding indices in $b$ and $c$. Add the resul back to $c$.

```c++
for (auto&& [{bx, by}, blocks] : mc::blocks(a)) {
  auto x_base = bx * block_width;
  auto y_base = by * block_height;
  for (auto i : __ranges::views::iota(I(0), I(block_height))) {
    for (auto j : __ranges::views::iota(I(0), I(block_width))) {
      if (0 == block[{i, j}]) continue;
      for (auto kk : __ranges::views::iota(I(0), I(k))) {
        auto x_addr = x_base + i;
        auto y_addr = y_base + j;
        I b_addr = col_addr * k + kk;
        I c_addr = col_addr * k + kk;
        C[c_addr] += block[{i, j}] * B[b_addr];
      }
    }
  }
}
```
