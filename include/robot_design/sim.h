#pragma once

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>
#include <memory>
#include <robot_design/robot.h>
#include <robot_design/types.h>

namespace robot_design {

class Simulation {
public:
  virtual void addRobot(std::shared_ptr<const Robot> robot) = 0;
  virtual std::shared_ptr<const Robot> getRobot(Index i) const = 0;
  virtual Index getRobotCount() const = 0;
  virtual void getTransform(Index item_id, Matrix4 *transform) const = 0;
  virtual void step(Scalar dt) = 0;
};

struct BulletRobotWrapper {
  BulletRobotWrapper(std::shared_ptr<const Robot> robot)
      : robot_(robot), multi_body_(nullptr), col_shapes_(), colliders_() {}
  std::shared_ptr<const Robot> robot_;
  std::shared_ptr<btMultiBody> multi_body_;
  std::vector<std::shared_ptr<btCollisionShape>> col_shapes_;
  std::vector<std::shared_ptr<btMultiBodyLinkCollider>> colliders_;
};

class BulletSimulation : public Simulation {
public:
  BulletSimulation(Scalar link_density, Scalar link_radius, Scalar friction);
  virtual ~BulletSimulation();
  BulletSimulation(const BulletSimulation &other) = delete;
  BulletSimulation &operator=(const BulletSimulation &other) = delete;
  virtual void addRobot(std::shared_ptr<const Robot>) override;
  virtual std::shared_ptr<const Robot> getRobot(Index i) const override;
  virtual Index getRobotCount() const override;
  virtual void getTransform(Index item_id, Matrix4 *transform) const override;
  virtual void step(Scalar dt) override;

private:
  void unregisterRobotWrapper(BulletRobotWrapper &wrapper);

  Scalar link_density_;  // Mass of links per unit of length
  Scalar link_radius_;
  Scalar friction_;
  Scalar internal_time_step_;
  std::shared_ptr<btDefaultCollisionConfiguration> collision_config_;
  std::shared_ptr<btHashedOverlappingPairCache> pair_cache_;
  std::shared_ptr<btCollisionDispatcher> dispatcher_;
  std::shared_ptr<btDbvtBroadphase> broadphase_;
  std::shared_ptr<btMultiBodyConstraintSolver> solver_;
  std::shared_ptr<btMultiBodyDynamicsWorld> world_;
  std::vector<BulletRobotWrapper> robot_wrappers_;
};

}  // namespace robot_design
