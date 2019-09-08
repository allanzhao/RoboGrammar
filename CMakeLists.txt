cmake_minimum_required(VERSION 3.0)

project(RobotDesign)

set(CMAKE_CXX_STANDARD 11)

set(ARGS_INCLUDE_DIRS extern/args)

# Use double precision in Bullet
add_compile_options(-DBT_USE_DOUBLE_PRECISION)
set(BULLET_ROOT extern/bullet3)
set(BULLET_INCLUDE_DIRS ${BULLET_ROOT}/src ${BULLET_ROOT}/Extras)
file(GLOB_RECURSE BULLET_SOURCES
    ${BULLET_ROOT}/src/BulletCollision/*.cpp
    ${BULLET_ROOT}/src/BulletDynamics/*.cpp
    ${BULLET_ROOT}/src/LinearMath/*.cpp
    ${BULLET_ROOT}/Extras/Serialize/BulletFileLoader/*.cpp
    ${BULLET_ROOT}/Extras/Serialize/BulletWorldImporter/*.cpp
)

add_compile_options(-DEIGEN_DONT_PARALLELIZE)
set(EIGEN_INCLUDE_DIRS extern/eigen)

find_package(GLEW 1.13 REQUIRED)
find_package(OpenGL REQUIRED)

set(GLFW_ROOT extern/glfw)
add_subdirectory(${GLFW_ROOT})

set(THREADPOOL_INCLUDE_DIRS extern/ThreadPool)

find_package(Torch REQUIRED)

include_directories(
    include
    ${ARGS_INCLUDE_DIRS}
    ${BULLET_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${THREADPOOL_INCLUDE_DIRS}
)

add_executable(RobotDesign
    src/articulated_robot.cpp
    src/grammar.cpp
    src/main.cpp
    src/optim.cpp
    src/render.cpp
    src/sim.cpp
    src/value.cpp
    ${BULLET_SOURCES}
)

target_link_libraries(RobotDesign PRIVATE
    glfw
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${TORCH_LIBRARIES}
)

if(MSVC)
  target_compile_options(RobotDesign PRIVATE /W4 /WX)
else()
  target_compile_options(RobotDesign PRIVATE -Wall -Werror)
  set_source_files_properties(
      ${BULLET_SOURCES} PROPERTIES COMPILE_FLAGS "-Wno-all -Wno-error"
  )
endif()

file(COPY data DESTINATION ${CMAKE_BINARY_DIR})
