#pragma once

#include <LinearMath/btQuaternion.h>
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

// Based on https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline std::size_t hashCombine(std::size_t seed) {
  return seed;
}

template <typename T, typename... Rest>
inline std::size_t hashCombine(std::size_t seed, const T &v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return hashCombine(seed, rest...);
}

}  // namespace robot_design
