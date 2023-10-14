#include <mc/mc.hpp>
#include <sycl/sycl.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {
  std::size_t m = 1000000;
  std::size_t k = 1000000;
  std::size_t nnz = 100000;

  using T = int;
  using I = int;

  sycl::queue q(sycl::gpu_selector_v);

  using allocator_type = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

  allocator_type alloc(q);

  // auto [values, rowptr, colind, shape, _] = mc::generate_csr<T, I,
  // allocator_type>(m, k, nnz, 0, alloc);
  /*
    mc::csr_matrix_view a(values.data(), rowptr.data(), colind.data(),
                             shape, nnz);
                             */

  auto [values, rowind, rowptr, colind, shape, num_rows, _] =
      mc::generate_dcsr<T, I, allocator_type>(m, k, nnz, 0, alloc);

  mc::dcsr_matrix_view a(values.data(), rowind.data(), rowptr.data(),
                         colind.data(), shape, num_rows, nnz);

  std::vector<T, allocator_type> b(k, 1, alloc);
  std::vector<T, allocator_type> c(m, 0, alloc);

  auto d_c = sycl::malloc_device<T>(m, q);
  q.memcpy(d_c, c.data(), sizeof(T) * m).wait();

  fmt::print("Before: {}\n", std::reduce(c.begin(), c.end()));

  std::span c_gpu(d_c, m);

  mc::spmv(q, c_gpu, a, b).wait();

  q.memcpy(c.data(), d_c, sizeof(T) * m).wait();

  fmt::print("After (GPU): {}\n", std::reduce(c.begin(), c.end()));

  std::vector<T> c_cpu(m, 0);

  mc::spmv(c_cpu, a, b);
  fmt::print("After (CPU): {}\n", std::reduce(c_cpu.begin(), c_cpu.end()));

  return 0;
}
