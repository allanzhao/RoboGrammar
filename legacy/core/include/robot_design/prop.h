#pragma once

#include <robot_design/types.h>

namespace robot_design {

enum class PropShape : Index { BOX, HEIGHTFIELD };

struct Prop {
  Prop() = default;
  Prop(PropShape shape, Scalar density, Scalar friction,
       const Vector3 &half_extents)
      : shape_(shape), density_(density), friction_(friction),
        half_extents_(half_extents) {}
  virtual ~Prop() {}

  // Shape
  PropShape shape_ = PropShape::BOX;
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

struct HeightfieldProp : Prop {
  template <typename T>
  HeightfieldProp(Scalar friction, const Vector3 &half_extents, T &&heightfield)
      : Prop(PropShape::HEIGHTFIELD, 0.0, friction, half_extents),
        heightfield_(std::forward<T>(heightfield)) {}
  virtual ~HeightfieldProp() {}

  // Heightfield data
  MatrixX heightfield_;
};

} // namespace robot_design
