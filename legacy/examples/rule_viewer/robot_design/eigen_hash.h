#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <robot_design/utils.h>

namespace std {

template <typename Scalar, int Rows, int Cols>
struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
  std::size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols> &mat) const {
    using robot_design::hashCombine;

    std::size_t seed = 0;
    for (int i = 0; i < mat.size(); ++i) {
      hashCombine(seed, mat.data()[i]);
    }
    return seed;
  }
};

template <typename Scalar>
struct hash<Eigen::Quaternion<Scalar>> {
  std::size_t operator()(const Eigen::Quaternion<Scalar> &q) const {
    using robot_design::hashCombine;

    std::size_t seed = 0;
    hashCombine(seed, q.x());
    hashCombine(seed, q.y());
    hashCombine(seed, q.z());
    hashCombine(seed, q.w());
    return seed;
  }
};

} // namespace std
