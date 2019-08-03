#pragma once

#include <robot_design/types.h>

namespace robot_design {

struct Prop {
  Prop(const Vector3 &initial_pos, const Quaternion &initial_rot,
       Scalar density, Scalar friction, const Vector3 &half_extents)
      : initial_pos_(initial_pos), initial_rot_(initial_rot),
        density_(density), friction_(friction), half_extents_(half_extents) {}

  // Initial position
  Vector3 initial_pos_;
  // Initial rotation
  Quaternion initial_rot_;
  // Density
  Scalar density_;
  // Friction
  Scalar friction_;
  // Half extents (size)
  Vector3 half_extents_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace drbs
