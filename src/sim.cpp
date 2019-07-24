#include <iostream>
#include <robot_design/sim.h>
#include <robot_design/utils.h>

namespace robot_design {

BulletSimulation::BulletSimulation(Scalar link_density, Scalar link_radius)
    : link_density_(link_density), link_radius_(link_radius),
      robot_wrappers_() {
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
  wrapper.col_shapes_.resize(robot->links_.size());

  for (int i = 0; i < robot->links_.size(); ++i) {
    const Link &link = robot->links_[i];

    auto col_shape = std::make_shared<btCapsuleShape>(link_radius_, link.length_);
    Scalar link_mass = link.length_ * link_density_;
    btVector3 link_inertia;
    col_shape->calculateLocalInertia(link_mass, link_inertia);

    if (i == 0) {
      assert(link.parent_ == -1 && link.joint_type_ == JointType::FREE);
      wrapper.multi_body_ = std::make_shared<btMultiBody>(
          /*n_links=*/robot->links_.size() - 1,
          /*mass=*/link_mass,
          /*inertia=*/link_inertia,
          /*fixedBase=*/false,
          /*canSleep=*/false);
      wrapper.multi_body_->setBaseWorldTransform(btTransform::getIdentity());
    } else {
      btQuaternion joint_rot(0, 0, 0, 1);  // TODO
      btVector3 joint_axis = bulletVector3FromEigen(link.joint_axis_);
      btVector3 joint_offset((link.joint_pos_ - 0.5) * robot->links_[link.parent_].length_, 0, 0);
      btVector3 com_offset(0.5 * link.length_, 0, 0);
      switch (link.joint_type_) {
      case JointType::HINGE:
        wrapper.multi_body_->setupRevolute(
          /*i=*/i - 1,  // Base link is already accounted for
          /*mass=*/link_mass,
          /*inertia=*/link_inertia,
          /*parent=*/link.parent_ - 1,
          /*rotParentToThis=*/joint_rot,
          /*joint_axis=*/joint_axis,
          /*parentComToThisPivotOffset=*/joint_offset,
          /*thisPivotToThisComOffset=*/com_offset,
          /*disableParentCollision=*/true);
        break;
      case JointType::FIXED:
        wrapper.multi_body_->setupFixed(
          /*i=*/i - 1,  // Base link is already accounted for
          /*mass=*/link_mass,
          /*inertia=*/link_inertia,
          /*parent=*/link.parent_ - 1,
          /*rotParentToThis=*/joint_rot,
          /*parentComToThisPivotOffset=*/joint_offset,
          /*thisPivotToThisComOffset=*/com_offset);
        break;
      }
    }

    wrapper.col_shapes_[i] = std::move(col_shape);
  }

  wrapper.multi_body_->finalizeMultiDof();
  world_->addMultiBody(wrapper.multi_body_.get());
  wrapper.multi_body_->setLinearDamping(0.0);
  wrapper.multi_body_->setAngularDamping(0.0);

  // Add collision shapes to world
  /* TODO
  wrapper.colliders_.resize(wrapper.col_shapes_.size());
  for (int i = 0; i < wrapper.col_shapes_.size(); ++i) {
    auto collider = std::make_shared<btMultiBodyLinkCollider>(wrapper.multi_body_, i - 1)
  */

  btAlignedObjectArray<btQuaternion> scratch_q;
  btAlignedObjectArray<btVector3> scratch_m;
  wrapper.multi_body_->forwardKinematics(scratch_q, scratch_m);
  btAlignedObjectArray<btQuaternion> world_to_local;
  btAlignedObjectArray<btVector3> local_origin;
  wrapper.multi_body_->updateCollisionObjectWorldTransforms(world_to_local, local_origin);
  for (int i = 0; i < robot->links_.size() - 1; ++i) {
    std::cout << "Link " << i << ":" << std::endl;
    const btVector3 &origin = wrapper.multi_body_->getLink(i).m_cachedWorldTransform.getOrigin();
    std::cout << origin.x() << ", " << origin.y() << ", " << origin.z() << std::endl;
  }
}

void BulletSimulation::getTransform(Index item_idx, Matrix4 *transform) const {

}

}  // namespace robot_design
