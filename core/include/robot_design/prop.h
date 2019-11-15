#pragma once

#include <robot_design/types.h>

namespace robot_design {

struct Prop {
  Prop(Scalar density, Scalar friction, const Vector3 &half_extents)
      : density_(density), friction_(friction), half_extents_(half_extents) {}

  // Density
  Scalar density_;
  // Friction
  Scalar friction_;
  // Half extents (size)
  Vector3 half_extents_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace drbs
