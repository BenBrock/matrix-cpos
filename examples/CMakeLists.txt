function(add_example example_name)
	add_executable(${example_name} ${example_name}.cpp)
	target_link_libraries(${example_name} mc)
endfunction()

add_example(dense_matrix_example)
add_example(csr_matrix_example)
