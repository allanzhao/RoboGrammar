#pragma once

#include <Eigen/Dense>
#include <robot_design/types.h>
#include <vector>

namespace robot_design {

struct Item {
  Item(const Matrix4 &initial_transform)
    : initial_transform_(initial_transform) {}
  Matrix4 initial_transform_;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct Box : public Item {
  Box(const Matrix4 &initial_transform, const Vector3 &half_size)
    : Item(initial_transform),
      half_size_(half_size) {}
  Vector3 half_size_;
};

struct Model {
  std::vector<Box> boxes_;
};

}  // namespace robot_design
