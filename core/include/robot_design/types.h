#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace robot_design {

using Index = int;
using Scalar = double;
using Vector3 = Eigen::Vector<Scalar, 3>;
using Vector4 = Eigen::Vector<Scalar, 4>;
using Vector6 = Eigen::Vector<Scalar, 6>;
using VectorX = Eigen::Vector<Scalar, Eigen::Dynamic>;
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Quaternion = Eigen::Quaternion<Scalar>;
using Affine3 = Eigen::Transform<Scalar, 3, Eigen::Affine>;
using Translation3 = Eigen::Translation<Scalar, 3>;
using Color = Eigen::Vector3f;

} // namespace robot_design
