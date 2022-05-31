#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <functional>
#include <robot_design/types.h>
#include <type_traits>
#include <vector>

namespace robot_design {

enum class LinkShape : Index { NONE, CAPSULE, CYLINDER };

enum class JointType : Index { NONE, FREE, HINGE, FIXED };

enum class JointControlMode : Index { POSITION, VELOCITY };

struct Link {
  Link(Index parent, JointType joint_type, Scalar joint_pos,
       const Quaternion &joint_rot, const Vector3 &joint_axis, LinkShape shape,
       Scalar length, Scalar radius, Scalar density, Scalar friction,
       Scalar joint_kp, Scalar joint_kd, Scalar joint_torque,
       JointControlMode joint_control_mode, const Color &color,
       const Color &joint_color, const std::string &label,
       const std::string &joint_label)
      : parent_(parent), joint_type_(joint_type), joint_pos_(joint_pos),
        joint_rot_(joint_rot), joint_axis_(joint_axis), shape_(shape),
        length_(length), radius_(radius), density_(density),
        friction_(friction), joint_kp_(joint_kp), joint_kd_(joint_kd),
        joint_torque_(joint_torque), joint_control_mode_(joint_control_mode),
        color_(color), joint_color_(joint_color), label_(label),
        joint_label_(joint_label) {}

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
  // Link radius
  Scalar radius_;
  // Link density
  Scalar density_; // Mass per unit of length
  // Link friction
  Scalar friction_;
  // Joint spring constant
  Scalar joint_kp_;
  // Joint damping coefficient
  Scalar joint_kd_;
  // Joint maximum torque
  Scalar joint_torque_;
  // Joint control mode
  JointControlMode joint_control_mode_;
  // Link color for rendering
  Color color_;
  // Joint color for rendering
  Color joint_color_;
  // Link label for rendering
  std::string label_;
  // Joint label for rendering
  std::string joint_label_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct Robot {
  std::vector<Link, Eigen::aligned_allocator<Link>> links_;
};

} // namespace robot_design

namespace std {

template <> struct hash<robot_design::LinkShape> {
  std::size_t operator()(const robot_design::LinkShape &link_shape) const {
    using type = typename std::underlying_type<robot_design::LinkShape>::type;
    return std::hash<type>()(static_cast<type>(link_shape));
  }
};

template <> struct hash<robot_design::JointType> {
  std::size_t operator()(const robot_design::JointType &joint_type) const {
    using type = typename std::underlying_type<robot_design::JointType>::type;
    return std::hash<type>()(static_cast<type>(joint_type));
  }
};

template <> struct hash<robot_design::JointControlMode> {
  std::size_t
  operator()(const robot_design::JointControlMode &joint_control_mode) const {
    using type =
        typename std::underlying_type<robot_design::JointControlMode>::type;
    return std::hash<type>()(static_cast<type>(joint_control_mode));
  }
};

} // namespace std
