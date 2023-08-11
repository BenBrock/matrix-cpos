enable_testing()

add_executable(
  mc-tests
  dense_matrix_view_test.cpp
)

target_link_libraries(mc-tests mc fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mc-tests)
