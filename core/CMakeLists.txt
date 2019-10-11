cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)

set(LIBRARY_NAME robotdesign)

set(BULLET_ROOT ${PROJECT_SOURCE_DIR}/extern/bullet3)
set(BULLET_INCLUDE_DIRS ${BULLET_ROOT}/src ${BULLET_ROOT}/Extras)
file(GLOB_RECURSE BULLET_SOURCES
    ${BULLET_ROOT}/src/BulletCollision/*.cpp
    ${BULLET_ROOT}/src/BulletDynamics/*.cpp
    ${BULLET_ROOT}/src/LinearMath/*.cpp
    ${BULLET_ROOT}/Extras/Serialize/BulletFileLoader/*.cpp
    ${BULLET_ROOT}/Extras/Serialize/BulletWorldImporter/*.cpp
)

add_compile_options(-DEIGEN_DONT_PARALLELIZE)
set(EIGEN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/extern/eigen)

find_package(GLEW 1.13 REQUIRED)
find_package(OpenGL REQUIRED)

set(GLFW_ROOT ${PROJECT_SOURCE_DIR}/extern/glfw)
add_subdirectory(${GLFW_ROOT} ${CMAKE_BINARY_DIR}/glfw)

set(PEGTL_ROOT ${PROJECT_SOURCE_DIR}/extern/PEGTL)
set(PEGTL_BUILD_TESTS OFF CACHE BOOL "Build PEGTL test programs")
set(PEGTL_BUILD_EXAMPLES OFF CACHE BOOL "Build PEGTL example programs")
add_subdirectory(${PEGTL_ROOT} ${CMAKE_BINARY_DIR}/PEGTL)

set(THREADPOOL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/extern/ThreadPool)

add_library(${LIBRARY_NAME}
    src/grammar.cpp
    src/graph.cpp
    src/optim.cpp
    src/render.cpp
    src/sim.cpp
    src/value.cpp
    ${BULLET_SOURCES}
)

target_include_directories(${LIBRARY_NAME} PUBLIC
    include
    ${BULLET_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${THREADPOOL_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${LIBRARY_NAME} PUBLIC
    glfw
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${TORCH_LIBRARIES}
)

target_link_libraries(${LIBRARY_NAME} PRIVATE
    pegtl
)

if(MSVC)
  target_compile_options(${LIBRARY_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${LIBRARY_NAME} PRIVATE -Wall -Werror)
  set_source_files_properties(
      ${BULLET_SOURCES} PROPERTIES COMPILE_FLAGS "-Wno-all -Wno-error"
  )
endif()
