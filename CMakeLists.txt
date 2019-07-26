cmake_minimum_required(VERSION 3.0)

project(RobotDesign)

set(CMAKE_CXX_STANDARD 11)

set(ARGS_INCLUDE_DIRS extern/args)

set(USE_DOUBLE_PRECISION ON)  # Use double precision in Bullet
set(BULLET_ROOT extern/bullet3)
set(BULLET_INCLUDE_DIRS ${BULLET_ROOT}/src)
file(GLOB_RECURSE BULLET_SOURCES
    ${BULLET_ROOT}/src/BulletCollision/*.cpp
    ${BULLET_ROOT}/src/BulletDynamics/*.cpp
    ${BULLET_ROOT}/src/LinearMath/*.cpp
)

set(EIGEN_INCLUDE_DIRS extern/eigen)

find_package(GLEW 1.13 REQUIRED)
find_package(OpenGL REQUIRED)

set(GLFW_ROOT extern/glfw)
add_subdirectory(${GLFW_ROOT})

include_directories(
    include
    ${ARGS_INCLUDE_DIRS}
    ${BULLET_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
)

add_executable(RobotDesign
    src/main.cpp
    src/render.cpp
    src/sim.cpp
    ${BULLET_SOURCES}
)

target_link_libraries(RobotDesign PRIVATE
    glfw
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

file(COPY data DESTINATION ${CMAKE_BINARY_DIR})
