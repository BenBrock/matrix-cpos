cmake_minimum_required(VERSION 3.20)
project(mc)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O3 -march=native")

find_package(fmt)

add_subdirectory(include)

add_subdirectory(examples)
