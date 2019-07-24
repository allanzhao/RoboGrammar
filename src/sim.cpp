#include <robot_design/sim.h>

namespace robot_design {

BulletSimulation::BulletSimulation() : robot_wrappers_() {
  collision_config_ = std::make_shared<btDefaultCollisionConfiguration>();
  dispatcher_ = std::make_shared<btCollisionDispatcher>(collision_config_.get());
  pair_cache_ = std::make_shared<btHashedOverlappingPairCache>();
  broadphase_ = std::make_shared<btDbvtBroadphase>(pair_cache_.get());
  solver_ = std::make_shared<btMultiBodyConstraintSolver>();
  world_ = std::make_shared<btMultiBodyDynamicsWorld>(dispatcher_.get(),
      broadphase_.get(), solver_.get(), collision_config_.get());
  world_->setGravity(btVector3(0, -9.81, 0));
}

void BulletSimulation::addRobot(std::shared_ptr<const Robot> robot) {
  robot_wrappers_.emplace_back(robot);
  BulletRobotWrapper &wrapper = robot_wrappers_.back();


}

void BulletSimulation::getTransform(Index item_idx, Matrix4 *transform) const {

}

}  // namespace robot_design
