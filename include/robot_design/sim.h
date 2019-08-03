#pragma once

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>
#include <memory>
#include <robot_design/prop.h>
#include <robot_design/robot.h>
#include <robot_design/types.h>

namespace robot_design {

class Simulation {
public:
  virtual void addRobot(std::shared_ptr<const Robot> robot) = 0;
  virtual void addProp(std::shared_ptr<const Prop> prop) = 0;
  virtual void removeRobot(std::shared_ptr<const Robot> robot) = 0;
  virtual void removeProp(std::shared_ptr<const Prop> prop) = 0;
  virtual std::shared_ptr<const Robot> getRobot(Index robot_idx) const = 0;
  virtual std::shared_ptr<const Prop> getProp(Index prop_idx) const = 0;
  virtual Index getRobotCount() const = 0;
  virtual Index getPropCount() const = 0;
  virtual void getLinkTransform(Index robot_idx, Index link_idx,
                            Matrix4 &transform) const = 0;
  virtual void getPropTransform(Index prop_idx, Matrix4 &transform) const = 0;
  virtual void advance(Scalar dt) = 0;
};

struct BulletRobotWrapper {
  BulletRobotWrapper(std::shared_ptr<const Robot> robot)
      : robot_(robot), multi_body_(), col_shapes_(), colliders_() {}
  std::shared_ptr<const Robot> robot_;
  std::shared_ptr<btMultiBody> multi_body_;
  std::vector<std::shared_ptr<btCollisionShape>> col_shapes_;
  std::vector<std::shared_ptr<btMultiBodyLinkCollider>> colliders_;
};

struct BulletPropWrapper {
  BulletPropWrapper(std::shared_ptr<const Prop> prop)
      : prop_(prop), rigid_body_(), col_shape_(), col_object_() {}
  std::shared_ptr<const Prop> prop_;
  std::shared_ptr<btRigidBody> rigid_body_;
  std::shared_ptr<btCollisionShape> col_shape_;
  std::shared_ptr<btCollisionObject> col_object_;
};

class BulletSimulation : public Simulation {
public:
  BulletSimulation();
  virtual ~BulletSimulation();
  BulletSimulation(const BulletSimulation &other) = delete;
  BulletSimulation &operator=(const BulletSimulation &other) = delete;
  virtual void addRobot(std::shared_ptr<const Robot> robot) override;
  virtual void addProp(std::shared_ptr<const Prop> prop) override;
  virtual void removeRobot(std::shared_ptr<const Robot> robot) override;
  virtual void removeProp(std::shared_ptr<const Prop> prop) override;
  virtual std::shared_ptr<const Robot> getRobot(Index robot_idx) const override;
  virtual std::shared_ptr<const Prop> getProp(Index prop_idx) const override;
  virtual Index getRobotCount() const override;
  virtual Index getPropCount() const override;
  virtual void getLinkTransform(Index robot_idx, Index link_idx,
                                Matrix4 &transform) const override;
  virtual void getPropTransform(Index prop_idx, Matrix4 &transform) const override;
  virtual void advance(Scalar dt) override;

private:
  void unregisterRobotWrapper(BulletRobotWrapper &robot_wrapper);
  void unregisterPropWrapper(BulletPropWrapper &prop_wrapper);

  Scalar internal_time_step_;
  std::shared_ptr<btDefaultCollisionConfiguration> collision_config_;
  std::shared_ptr<btHashedOverlappingPairCache> pair_cache_;
  std::shared_ptr<btCollisionDispatcher> dispatcher_;
  std::shared_ptr<btDbvtBroadphase> broadphase_;
  std::shared_ptr<btMultiBodyConstraintSolver> solver_;
  std::shared_ptr<btMultiBodyDynamicsWorld> world_;
  std::vector<BulletRobotWrapper> robot_wrappers_;
  std::vector<BulletPropWrapper> prop_wrappers_;
};

}  // namespace robot_design
