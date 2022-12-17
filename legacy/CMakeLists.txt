cmake_minimum_required(VERSION 3.0)

project(RobotDesign)

# Use double precision in Bullet
add_compile_options(-DBT_USE_DOUBLE_PRECISION)

add_subdirectory(core)
add_subdirectory(examples/python_bindings)
add_subdirectory(examples/rule_viewer)

file(COPY ${PROJECT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
