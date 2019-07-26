#pragma once

#include <fstream>
#include <LinearMath/btQuaternion.h>
#include <LinearMath/btVector3.h>
#include <robot_design/types.h>
#include <sstream>
#include <string>

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

inline std::string loadString(const std::string &path) {
  std::ifstream ifs(path);
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

}  // namespace robot_design
