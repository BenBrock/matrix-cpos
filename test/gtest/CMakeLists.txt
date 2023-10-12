enable_testing()

add_executable(
  mc-tests
  dense_matrix_view_test.cpp
  csr_matrix_view_test.cpp
  csc_matrix_view_test.cpp
  coo_matrix_view_test.cpp
  mdspan_test.cpp
)

target_link_libraries(mc-tests mc fmt mdspan GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mc-tests)
