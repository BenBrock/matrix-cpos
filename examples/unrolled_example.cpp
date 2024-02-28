#include <chrono>
#include <fmt/ranges.h>
#include <mc/mc.hpp>

int main(int argc, char** argv) {
  using T = float;

  std::size_t size = 1024 * 1024 * 1024 + (lrand48() % 10);

  std::vector<float> vec(size);

  for (std::size_t i = 0; i < size; i++) {
    vec[i] = lrand48() % 100;
  }

  auto begin = std::chrono::high_resolution_clock::now();

  float sum = 0;

  std::for_each(vec.begin(), vec.end(), [&](auto v) { sum += v; });

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double normal_duration = duration;

  fmt::print("Normal sum {} in {} ms\n", sum, duration * 1000);

  begin = std::chrono::high_resolution_clock::now();

  sum = 0;

  mc::unrolled_for_each(32_i, vec, [&](auto v) { sum += v; });

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();

  fmt::print("Unrolled sum {} in {} ms\n", sum, duration * 1000);

  fmt::print("{} speedup\n", normal_duration / duration);

  return 0;
}
