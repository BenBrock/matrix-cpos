
add_library(mc INTERFACE)

find_package(fmt)
target_link_libraries(mc INTERFACE fmt)

target_include_directories(mc INTERFACE .)
