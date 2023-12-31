cmake_minimum_required(VERSION 3.20)
project(mc)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -march=native")

option(ENABLE_SYCL "Build with SYCL" OFF)
option(GCC_TOOLCHAIN "Path to GCC installation to be used by Clang-based compilers" OFF)

if(ENABLE_SYCL)
    add_compile_options(-fsycl)
    add_link_options(-fsycl)
endif()

if(GCC_TOOLCHAIN)
  add_compile_options(--gcc-toolchain=${GCC_TOOLCHAIN})
  add_link_options(--gcc-toolchain=${GCC_TOOLCHAIN})
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG mdspan-0.6.0)
FetchContent_MakeAvailable(mdspan)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.1.1)
FetchContent_MakeAvailable(fmt)

add_subdirectory(include)
add_subdirectory(examples)
add_subdirectory(test)
