cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
find_package(OpenGL REQUIRED COMPONENTS OpenGL)

set(GLFW_ROOT ${PROJECT_SOURCE_DIR}/extern/glfw)
add_subdirectory(${GLFW_ROOT} ${CMAKE_BINARY_DIR}/glfw)

set(LODEPNG_ROOT ${PROJECT_SOURCE_DIR}/extern/lodepng)
set(LODEPNG_INCLUDE_DIRS ${LODEPNG_ROOT})
set(LODEPNG_SOURCES ${LODEPNG_ROOT}/lodepng.cpp)

set(PEGTL_ROOT ${PROJECT_SOURCE_DIR}/extern/PEGTL)
set(PEGTL_BUILD_TESTS OFF CACHE BOOL "Build PEGTL test programs")
set(PEGTL_BUILD_EXAMPLES OFF CACHE BOOL "Build PEGTL example programs")
add_subdirectory(${PEGTL_ROOT} ${CMAKE_BINARY_DIR}/PEGTL)

set(THREADPOOL_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/extern/ThreadPool)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build TinyXML2 shared libraries")
set(BUILD_TESTS OFF CACHE BOOL "Build TinyXML2 test programs")
set(TINYXML2_ROOT ${PROJECT_SOURCE_DIR}/extern/tinyxml2)
add_subdirectory(${TINYXML2_ROOT} ${CMAKE_BINARY_DIR}/tinyxml2)

set(ROBOTDESIGN_SOURCES
    src/bitmap_font.cpp
    src/gl_renderer.cpp
    src/glfw_viewer.cpp
    src/graph_build.cpp
    src/graph_io.cpp
    src/graph_rewrite.cpp
    src/optim.cpp
    src/render.cpp
    src/sim.cpp
)

add_library(${LIBRARY_NAME}
    ${BULLET_SOURCES}
    ${LODEPNG_SOURCES}
    ${ROBOTDESIGN_SOURCES}
)

target_include_directories(${LIBRARY_NAME} PUBLIC
    include
    ${BULLET_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${THREADPOOL_INCLUDE_DIRS}
)

target_include_directories(${LIBRARY_NAME} PRIVATE
    ${LODEPNG_INCLUDE_DIRS}
    ${TINYXML2_INCLUDE_DIRS}
)

target_link_libraries(${LIBRARY_NAME} PUBLIC
    OpenGL::GL
    glfw
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
)

target_link_libraries(${LIBRARY_NAME} PRIVATE
    pegtl
    tinyxml2
)

if(MSVC)
  target_compile_options(${LIBRARY_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${LIBRARY_NAME} PRIVATE -Wall -Werror)
  set_source_files_properties(
      ${BULLET_SOURCES} PROPERTIES COMPILE_FLAGS "-Wno-all -Wno-error"
  )
endif()
