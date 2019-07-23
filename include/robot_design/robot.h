#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <robot_design/types.h>
#include <vector>

namespace robot_design {

enum class JointType : Index { FREE, HINGE };

enum class ShapeType : Index { BOX };

struct Joint {
  Joint(Index parent, JointType type, const Vector3 &pos, const Vector4 &rot)
      : parent_(parent), type_(type), pos_(pos), rot_(rot) {}

  // Parent joint (-1 for root joints)
  Index parent_;
  // Joint type
  JointType type_;
  // Position relative to the parent joint
  Vector3 pos_;
  // Rotation relative to the parent joint
  Vector4 rot_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Body {
  Body(Index joint, const Vector3 &pos, const Vector4 &rot, Scalar mass,
      const Vector3 &inertia)
      : joint_(joint), pos_(pos), rot_(rot), mass_(mass), inertia_(inertia) {}

  // Parent joint
  Index joint_;
  // Position relative to the parent joint
  Vector3 pos_;
  // Rotation relative to the parent joint
  Vector4 rot_;
  // Mass
  Scalar mass_;
  // Inertia matrix diagonal entries
  Vector3 inertia_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Shape {
  Shape(Index body, ShapeType type, const Vector3 &pos, const Vector4 &rot,
      const Vector3 &half_size)
      : body_(body), type_(type), pos_(pos), rot_(rot), half_size_(half_size) {}

  // Parent body
  Index body_;
  // Shape type
  ShapeType type_;
  // Position relative to the parent body
  Vector3 pos_;
  // Rotation relative to the parent body
  Vector4 rot_;
  // Half the size in each axis
  Vector3 half_size_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Robot {
  std::vector<Joint, Eigen::aligned_allocator<Joint>> joints_;
  std::vector<Body, Eigen::aligned_allocator<Body>> bodies_;
  std::vector<Shape, Eigen::aligned_allocator<Shape>> shapes_;
};

}  // namespace drbs
