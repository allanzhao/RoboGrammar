cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_NAME Viewer)

set(ARGS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/extern/args)

add_compile_options(-DEIGEN_DONT_PARALLELIZE)
set(EIGEN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/extern/eigen)

add_executable(${EXECUTABLE_NAME}
    main.cpp
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE
    ${ARGS_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE
    robotdesign
)

if(MSVC)
  target_compile_options(${EXECUTABLE_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${EXECUTABLE_NAME} PRIVATE -Wall -Werror)
endif()
