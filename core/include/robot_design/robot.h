#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <robot_design/types.h>
#include <vector>

namespace robot_design {

enum class LinkShape : Index { NONE, CAPSULE, CYLINDER };

enum class JointType : Index { FREE, HINGE, FIXED };

struct Link {
  Link(Index parent, JointType joint_type, Scalar joint_pos,
       const Quaternion &joint_rot, const Vector3 &joint_axis, LinkShape shape,
       Scalar length, const Color &color, const Color &joint_color)
      : parent_(parent), joint_type_(joint_type), joint_pos_(joint_pos),
        joint_rot_(joint_rot), joint_axis_(joint_axis), shape_(shape),
        length_(length), color_(color), joint_color_(joint_color) {}

  // Parent link index (-1 for base link)
  Index parent_;
  // Joint type
  JointType joint_type_;
  // Joint position on parent link (0 = beginning, 1 = end)
  Scalar joint_pos_;
  // Joint rotation relative to parent link
  Quaternion joint_rot_;
  // Joint axis relative to the joint frame (defined by previous 3 parameters)
  Vector3 joint_axis_;
  // Link shape
  LinkShape shape_;
  // Link length
  Scalar length_;
  // Link color for rendering
  Color color_;
  // Joint color for rendering
  Color joint_color_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Robot {
  Robot(Scalar link_density, Scalar link_radius, Scalar friction,
        Scalar motor_kp, Scalar motor_kd)
      : link_density_(link_density), link_radius_(link_radius),
        friction_(friction), motor_kp_(motor_kp), motor_kd_(motor_kd),
        links_() {}
  Scalar link_density_; // Mass of links per unit of length
  Scalar link_radius_;
  Scalar friction_;
  Scalar motor_kp_;
  Scalar motor_kd_;
  std::vector<Link, Eigen::aligned_allocator<Link>> links_;
};

} // namespace drbs
