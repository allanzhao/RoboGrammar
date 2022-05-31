cmake_minimum_required(VERSION 3.0)

project(RuleViewer)

set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_NAME RuleViewer)

set(ARGS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/extern/args)

add_compile_options(-DEIGEN_DONT_PARALLELIZE)
set(EIGEN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/extern/eigen)

set(LODEPNG_ROOT ${PROJECT_SOURCE_DIR}/extern/lodepng)
set(LODEPNG_INCLUDE_DIRS ${LODEPNG_ROOT})
set(LODEPNG_SOURCES ${LODEPNG_ROOT}/lodepng.cpp)

add_executable(${EXECUTABLE_NAME}
    main.cpp
    ${LODEPNG_SOURCES}
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE
    ${ARGS_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${LODEPNG_INCLUDE_DIRS}
)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE
    robotdesign
)

if(MSVC)
  target_compile_options(${EXECUTABLE_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${EXECUTABLE_NAME} PRIVATE -Wall -Werror)
endif()
