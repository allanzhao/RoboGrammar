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
       Scalar length, const Color &color, const Color &joint_color,
       const std::string &label, const std::string &joint_label)
      : parent_(parent), joint_type_(joint_type), joint_pos_(joint_pos),
        joint_rot_(joint_rot), joint_axis_(joint_axis), shape_(shape),
        length_(length), color_(color), joint_color_(joint_color),
        label_(label), joint_label_(joint_label) {}

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
  Robot() = default;
  Robot(Scalar link_density, Scalar link_radius, Scalar friction,
        Scalar motor_kp, Scalar motor_kd)
      : link_density_(link_density), link_radius_(link_radius),
        friction_(friction), motor_kp_(motor_kp), motor_kd_(motor_kd),
        links_() {}

  Scalar link_density_ = 1.0; // Mass of links per unit of length
  Scalar link_radius_ = 0.05;
  Scalar friction_ = 0.9;
  Scalar motor_kp_ = 2.0;
  Scalar motor_kd_ = 0.1;
  std::vector<Link, Eigen::aligned_allocator<Link>> links_;
};

} // namespace drbs

namespace std {

template <>
struct hash<robot_design::LinkShape> {
  std::size_t operator()(const robot_design::LinkShape &link_shape) const {
    using type = typename std::underlying_type<robot_design::LinkShape>::type;
    return std::hash<type>()(static_cast<type>(link_shape));
  }
};

template <>
struct hash<robot_design::JointType> {
  std::size_t operator()(const robot_design::JointType &joint_type) const {
    using type = typename std::underlying_type<robot_design::JointType>::type;
    return std::hash<type>()(static_cast<type>(joint_type));
  }
};

} // namespace std
