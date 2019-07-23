#pragma once

#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <memory>
#include <robot_design/robot.h>
#include <robot_design/types.h>

namespace robot_design {

class Simulation {
  virtual void addRobot(std::shared_ptr<const Robot> robot) = 0;
  virtual void getTransform(Index item_id, Matrix4 *transform) const = 0;
};

struct BulletRobotWrapper {
  BulletRobotWrapper(std::shared_ptr<const Robot> robot)
      : robot_(robot), multi_body_(nullptr) {}
  std::shared_ptr<const Robot> robot_;
  std::shared_ptr<btMultiBody> multi_body_;
};

class BulletSimulation : public Simulation {
public:
  BulletSimulation();
  virtual void addRobot(std::shared_ptr<const Robot>) override;
  virtual void getTransform(Index item_id, Matrix4 *transform) const override;

private:
  std::shared_ptr<btDefaultCollisionConfiguration> collision_config_;
  std::shared_ptr<btHashedOverlappingPairCache> pair_cache_;
  std::shared_ptr<btCollisionDispatcher> dispatcher_;
  std::shared_ptr<btDbvtBroadphase> broadphase_;
  std::shared_ptr<btMultiBodyConstraintSolver> solver_;
  std::shared_ptr<btMultiBodyDynamicsWorld> world_;
  std::vector<BulletRobotWrapper> robot_wrappers_;
};

}  // namespace robot_design
