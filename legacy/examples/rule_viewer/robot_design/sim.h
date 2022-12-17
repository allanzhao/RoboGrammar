#pragma once

#include <BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h>
#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointMotor.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>
#include <Serialize/BulletFileLoader/btBulletFile.h>
#include <btBulletDynamicsCommon.h>
#include <memory>
#include <robot_design/prop.h>
#include <robot_design/robot.h>
#include <robot_design/types.h>

namespace robot_design {

using Eigen::Ref;

class Simulation {
public:
  virtual ~Simulation() {}
  virtual Index addRobot(std::shared_ptr<const Robot> robot, const Vector3 &pos,
                         const Quaternion &rot) = 0;
  virtual Index addProp(std::shared_ptr<const Prop> prop, const Vector3 &pos,
                        const Quaternion &rot) = 0;
  virtual void removeRobot(Index robot_idx) = 0;
  virtual void removeProp(Index prop_idx) = 0;
  virtual std::shared_ptr<const Robot> getRobot(Index robot_idx) const = 0;
  virtual std::shared_ptr<const Prop> getProp(Index prop_idx) const = 0;
  virtual Index getRobotCount() const = 0;
  virtual Index getPropCount() const = 0;
  virtual Index findRobotIndex(const Robot &robot) const = 0;
  virtual Index findPropIndex(const Prop &prop) const = 0;
  virtual void getLinkTransform(Index robot_idx, Index link_idx,
                                Ref<Matrix4> transform) const = 0;
  virtual void getPropTransform(Index prop_idx,
                                Ref<Matrix4> transform) const = 0;
  virtual void getLinkVelocity(Index robot_idx, Index link_idx,
                               Ref<Vector6> vel) const = 0;
  virtual Scalar getLinkMass(Index robot_idx, Index link_idx) const = 0;
  virtual int getRobotDofCount(Index robot_idx) const = 0;
  virtual void getJointPositions(Index robot_idx, Ref<VectorX> pos) const = 0;
  virtual void getJointVelocities(Index robot_idx, Ref<VectorX> vel) const = 0;
  virtual void getJointTargetPositions(Index robot_idx,
                                       Ref<VectorX> target_pos) const = 0;
  virtual void getJointTargetVelocities(Index robot_idx,
                                        Ref<VectorX> target_vel) const = 0;
  virtual void getJointMotorTorques(Index robot_idx,
                                    Ref<VectorX> motor_torques) const = 0;
  virtual void setJointTargets(Index robot_idx,
                               const Ref<const VectorX> &target) = 0;
  virtual void
  setJointTargetPositions(Index robot_idx,
                          const Ref<const VectorX> &target_pos) = 0;
  virtual void
  setJointTargetVelocities(Index robot_idx,
                           const Ref<const VectorX> &target_vel) = 0;
  virtual void addJointTorques(Index robot_idx,
                               const Ref<const VectorX> &torque) = 0;
  virtual void addLinkForceTorque(Index robot_idx, Index link_idx,
                                  const Ref<const Vector3> &force,
                                  const Ref<const Vector3> &torque) = 0;
  virtual void getRobotWorldAABB(Index robot_idx, Ref<Vector3> lower,
                                 Ref<Vector3> upper) const = 0;
  virtual bool robotHasCollision(Index robot_idx) const = 0;
  virtual Scalar getTimeStep() const = 0;
  virtual Vector3 getGravity() const = 0;
  virtual void setGravity(const Ref<const Vector3> &gravity) = 0;
  virtual void saveState() = 0;
  virtual void restoreState() = 0;
  virtual void step() = 0;
};

struct BulletRobotWrapper {
  BulletRobotWrapper(std::shared_ptr<const Robot> robot) : robot_(robot) {}
  std::shared_ptr<const Robot> robot_;
  std::shared_ptr<btMultiBody> multi_body_;
  std::vector<std::shared_ptr<btCollisionShape>> col_shapes_;
  std::vector<std::shared_ptr<btMultiBodyLinkCollider>> colliders_;
  std::vector<std::shared_ptr<btMultiBodyJointMotor>> motors_;
  VectorX joint_kp_;
  VectorX joint_kd_;
  VectorX joint_target_pos_;
  VectorX joint_target_vel_;
  VectorX joint_motor_torques_;
};

struct BulletPropWrapper {
  BulletPropWrapper(std::shared_ptr<const Prop> prop)
      : prop_(prop), rigid_body_(), col_shape_(), col_object_() {}
  std::shared_ptr<const Prop> prop_;
  std::shared_ptr<btRigidBody> rigid_body_;
  std::shared_ptr<btCollisionShape> col_shape_;
  std::shared_ptr<btCollisionObject> col_object_;
};

struct BulletSavedState {
  BulletSavedState() {}
  BulletSavedState(std::shared_ptr<btSerializer> serializer,
                   std::shared_ptr<bParse::btBulletFile> bullet_file)
      : serializer_(std::move(serializer)),
        bullet_file_(std::move(bullet_file)) {}
  std::shared_ptr<btSerializer> serializer_;
  std::shared_ptr<bParse::btBulletFile> bullet_file_;
};

class BulletSimulation : public Simulation {
public:
  BulletSimulation(Scalar time_step = 1.0 / 240);
  virtual ~BulletSimulation();
  BulletSimulation(const BulletSimulation &other) = delete;
  BulletSimulation &operator=(const BulletSimulation &other) = delete;
  virtual Index addRobot(std::shared_ptr<const Robot> robot, const Vector3 &pos,
                         const Quaternion &rot) override;
  virtual Index addProp(std::shared_ptr<const Prop> prop, const Vector3 &pos,
                        const Quaternion &rot) override;
  virtual void removeRobot(Index robot_idx) override;
  virtual void removeProp(Index prop_idx) override;
  virtual std::shared_ptr<const Robot> getRobot(Index robot_idx) const override;
  virtual std::shared_ptr<const Prop> getProp(Index prop_idx) const override;
  virtual Index getRobotCount() const override;
  virtual Index getPropCount() const override;
  virtual Index findRobotIndex(const Robot &robot) const override;
  virtual Index findPropIndex(const Prop &prop) const override;
  virtual void getLinkTransform(Index robot_idx, Index link_idx,
                                Ref<Matrix4> transform) const override;
  virtual void getPropTransform(Index prop_idx,
                                Ref<Matrix4> transform) const override;
  virtual void getLinkVelocity(Index robot_idx, Index link_idx,
                               Ref<Vector6> vel) const override;
  virtual Scalar getLinkMass(Index robot_idx, Index link_idx) const override;
  virtual int getRobotDofCount(Index robot_idx) const override;
  virtual void getJointPositions(Index robot_idx,
                                 Ref<VectorX> pos) const override;
  virtual void getJointVelocities(Index robot_idx,
                                  Ref<VectorX> vel) const override;
  virtual void getJointTargetPositions(Index robot_idx,
                                       Ref<VectorX> target_pos) const override;
  virtual void getJointTargetVelocities(Index robot_idx,
                                        Ref<VectorX> target_vel) const override;
  virtual void getJointMotorTorques(Index robot_idx,
                                    Ref<VectorX> motor_torques) const override;
  virtual void setJointTargets(Index robot_idx,
                               const Ref<const VectorX> &target) override;
  virtual void
  setJointTargetPositions(Index robot_idx,
                          const Ref<const VectorX> &target_pos) override;
  virtual void
  setJointTargetVelocities(Index robot_idx,
                           const Ref<const VectorX> &target_vel) override;
  virtual void addJointTorques(Index robot_idx,
                               const Ref<const VectorX> &torque) override;
  virtual void addLinkForceTorque(Index robot_idx, Index link_idx,
                                  const Ref<const Vector3> &force,
                                  const Ref<const Vector3> &torque) override;
  virtual void getRobotWorldAABB(Index robot_idx, Ref<Vector3> lower,
                                 Ref<Vector3> upper) const override;
  virtual bool robotHasCollision(Index robot_idx) const override;
  virtual Scalar getTimeStep() const override;
  virtual Vector3 getGravity() const override;
  virtual void setGravity(const Ref<const Vector3> &gravity) override;
  virtual void saveState() override;
  virtual void restoreState() override;
  virtual void step() override;

private:
  struct OverlapFilterCallback : public btOverlapFilterCallback {
    virtual bool
    needBroadphaseCollision(btBroadphaseProxy *proxy0,
                            btBroadphaseProxy *proxy1) const override;
  };

  void unregisterRobotWrapper(BulletRobotWrapper &robot_wrapper);
  void unregisterPropWrapper(BulletPropWrapper &prop_wrapper);

  Scalar time_step_;
  OverlapFilterCallback overlap_filter_callback_;
  std::shared_ptr<btDefaultCollisionConfiguration> collision_config_;
  std::shared_ptr<btHashedOverlappingPairCache> pair_cache_;
  std::shared_ptr<btCollisionDispatcher> dispatcher_;
  std::shared_ptr<btDbvtBroadphase> broadphase_;
  std::shared_ptr<btMultiBodyConstraintSolver> solver_;
  std::shared_ptr<btMultiBodyDynamicsWorld> world_;
  std::vector<BulletRobotWrapper> robot_wrappers_;
  std::vector<BulletPropWrapper> prop_wrappers_;
  BulletSavedState saved_state_;
};

} // namespace robot_design
