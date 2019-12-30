#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <functional>
#include <robot_design/types.h>
#include <type_traits>
#include <vector>

namespace robot_design {

enum class LinkShape : Index { NONE, CAPSULE, CYLINDER };

enum class JointType : Index { FREE, HINGE, FIXED };

struct Link {
  Link() = default;
  Link(Index parent, JointType joint_type, Scalar joint_pos,
       const Quaternion &joint_rot, const Vector3 &joint_axis, LinkShape shape,
       Scalar length, Scalar radius, Scalar density, Scalar friction,
       Scalar joint_kp, Scalar joint_kd, const Color &color,
       const Color &joint_color, const std::string &label,
       const std::string &joint_label)
      : parent_(parent), joint_type_(joint_type), joint_pos_(joint_pos),
        joint_rot_(joint_rot), joint_axis_(joint_axis), shape_(shape),
        length_(length), radius_(radius), density_(density),
        friction_(friction), joint_kp_(joint_kp), joint_kd_(joint_kd),
        color_(color), joint_color_(joint_color), label_(label),
        joint_label_(joint_label) {}

  // Parent link index (-1 for base link)
  Index parent_ = -1;
  // Joint type
  JointType joint_type_ = JointType::FIXED;
  // Joint position on parent link (0 = beginning, 1 = end)
  Scalar joint_pos_ = 1.0;
  // Joint rotation relative to parent link
  Quaternion joint_rot_ = Quaternion::Identity();
  // Joint axis relative to the joint frame (defined by previous 3 parameters)
  Vector3 joint_axis_ = Vector3::UnitZ();
  // Link shape
  LinkShape shape_ = LinkShape::CAPSULE;
  // Link length
  Scalar length_ = 1.0;
  // Link radius
  Scalar radius_ = 0.05;
  // Link density
  Scalar density_ = 0.5; // Mass per unit of length
  // Link friction
  Scalar friction_ = 0.9;
  // Joint spring constant
  Scalar joint_kp_ = 2.0;
  // Joint damping coefficient
  Scalar joint_kd_ = 0.1;
  // Link color for rendering
  Color color_ = {0.45f, 0.5f, 0.55f}; // Slate gray
  // Joint color for rendering
  Color joint_color_ = {1.0f, 0.5f, 0.3f}; // Coral
  // Link label for rendering
  std::string label_;
  // Joint label for rendering
  std::string joint_label_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct Robot {
  std::vector<Link, Eigen::aligned_allocator<Link>> links_;
};

} // namespace drbs

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

} // namespace std
