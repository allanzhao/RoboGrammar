#pragma once

#include <robot_design/types.h>

namespace robot_design {

struct Prop {
  Prop() = default;
  Prop(Scalar density, Scalar friction, const Vector3 &half_extents)
      : density_(density), friction_(friction), half_extents_(half_extents) {}

  // Density
  Scalar density_ = 0.0;
  // Friction
  Scalar friction_ = 0.9;
  // Half extents (size)
  Vector3 half_extents_ = Vector3::Constant(1.0);
  // Color for rendering
  Color color_ = {0.8f, 0.7f, 0.6f}; // Tan

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

} // namespace drbs
