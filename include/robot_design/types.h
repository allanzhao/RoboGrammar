#pragma once

#include <Eigen/Dense>

namespace robot_design {

using Index = int;
using Scalar = double;
using Vector3 = Eigen::Vector<Scalar, 3>;
using Vector4 = Eigen::Vector<Scalar, 4>;
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;

}  // namespace robot_design
