#pragma once

#include <cstddef>
#include <functional>
#include <LinearMath/btMatrix3x3.h>
#include <LinearMath/btQuaternion.h>
#include <LinearMath/btTransform.h>
#include <LinearMath/btVector3.h>
#include <robot_design/types.h>

namespace robot_design {

inline btVector3 bulletVector3FromEigen(const Vector3 &v) {
  return btVector3(v.x(), v.y(), v.z());
}

inline Vector3 eigenVector3FromBullet(const btVector3 &v) {
  return Vector3(v.x(), v.y(), v.z());
}

inline btQuaternion bulletQuaternionFromEigen(const Quaternion &q) {
  return btQuaternion(q.x(), q.y(), q.z(), q.w());
}

inline Quaternion eigenQuaternionFromBullet(const btQuaternion &q) {
  return Quaternion(q.w(), q.x(), q.y(), q.z());
}

inline btMatrix3x3 bulletMatrix3x3FromEigen(const Matrix3 &m) {
  // clang-format off
  return btMatrix3x3(m(0, 0), m(0, 1), m(0, 2),
                     m(1, 0), m(1, 1), m(1, 2),
                     m(2, 0), m(2, 1), m(2, 2));
  // clang-format on
}

inline Matrix3 eigenMatrix3FromBullet(const btMatrix3x3 &m) {
  Matrix3 result;
  // clang-format off
  result << m[0][0], m[0][1], m[0][2],
            m[1][0], m[1][1], m[1][2],
            m[2][0], m[2][1], m[2][2];
  // clang-format on
  return result;
}

inline btTransform bulletTransformFromEigen(const Matrix4 &m) {
  return btTransform(bulletMatrix3x3FromEigen(m.topLeftCorner<3, 3>()),
                     bulletVector3FromEigen(m.topRightCorner<3, 1>()));
}

inline Matrix4 eigenMatrix4FromBullet(const btTransform &t) {
  Matrix4 result;
  result << eigenMatrix3FromBullet(t.getBasis()),
      eigenVector3FromBullet(t.getOrigin()), 0, 0, 0, 1;
  return result;
}

template <typename T> inline T clamp(T val, T lower, T upper) {
  return std::max(lower, std::min(upper, val));
}

// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x, originally from Boost
template <typename T>
inline void hashCombine(std::size_t &seed, const T &v) {
  std::hash<T> hash_fn;
  seed ^= hash_fn(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace robot_design
