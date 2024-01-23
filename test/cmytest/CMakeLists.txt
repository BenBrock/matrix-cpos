add_executable(
  cmy-tests
  bcsr_matrix_view_test.cpp
)

target_link_libraries(cmy-tests mc fmt mdspan)

