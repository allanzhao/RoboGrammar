#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>
#include <Serialize/BulletWorldImporter/btMultiBodyWorldImporter.h>
#include <cstddef>
#include <limits>
#include <robot_design/prop.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <robot_design/utils.h>
#include <stdexcept>

namespace robot_design {

BulletSimulation::BulletSimulation(Scalar time_step) : time_step_(time_step) {
    collision_config_ = std::make_shared<btDefaultCollisionConfiguration>();
    dispatcher_ =
            std::make_shared<btCollisionDispatcher>(collision_config_.get());
    pair_cache_ = std::make_shared<btHashedOverlappingPairCache>();
    pair_cache_->setOverlapFilterCallback(&overlap_filter_callback_);
    broadphase_ = std::make_shared<btDbvtBroadphase>(pair_cache_.get());
    solver_ = std::make_shared<btMultiBodyConstraintSolver>();
    world_ = std::make_shared<btMultiBodyDynamicsWorld>(
            dispatcher_.get(), broadphase_.get(), solver_.get(),
            collision_config_.get());
    world_->setGravity(btVector3(0, -9.81, 0));
    world_->getDispatchInfo().m_deterministicOverlappingPairs = true;
}

BulletSimulation::~BulletSimulation() {
    for (auto &robot_wrapper : robot_wrappers_) {
        unregisterRobotWrapper(robot_wrapper);
    }
    for (auto &prop_wrapper : prop_wrappers_) {
        unregisterPropWrapper(prop_wrapper);
    }
}

Index BulletSimulation::addRobot(std::shared_ptr<const Robot> robot, const Vector3 &pos, const Quaternion &rot) {
    robot_wrappers_.emplace_back(robot);
    BulletRobotWrapper &wrapper = robot_wrappers_.back();
    wrapper.col_shapes_.resize(robot->links_.size());

    for (std::size_t i = 0; i < robot->links_.size(); ++i) {
        const Link &link = robot->links_[i];

        std::shared_ptr<btCollisionShape> col_shape;
        switch (link.shape_) {
        case LinkShape::CAPSULE:
            col_shape = std::make_shared<btCapsuleShapeX>(link.radius_, link.length_);
            break;
        case LinkShape::CYLINDER:
            col_shape = std::make_shared<btCylinderShapeX>(
                    btVector3{0.5 * link.length_, link.radius_, link.radius_});
            break;
        default:
            throw std::runtime_error("Unexpected link shape");
        }
        Scalar link_mass = link.length_ * link.density_;
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
            wrapper.multi_body_->setBaseWorldTransform(btTransform(
                    /*q=*/bulletQuaternionFromEigen(rot),
                    /*c=*/bulletVector3FromEigen(pos)));
        } else {
            btQuaternion joint_rot = bulletQuaternionFromEigen(link.joint_rot_);
            btVector3 joint_axis = bulletVector3FromEigen(link.joint_axis_);
            btVector3 joint_offset(
                    (link.joint_pos_ - 0.5) * robot->links_[link.parent_].length_, 0, 0);
            btVector3 com_offset(0.5 * link.length_, 0, 0);
            switch (link.joint_type_) {
            case JointType::HINGE:
                wrapper.multi_body_->setupRevolute(
                        /*i=*/i - 1, // Base link is already accounted for
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
                        /*i=*/i - 1, // Base link is already accounted for
                        /*mass=*/link_mass,
                        /*inertia=*/link_inertia,
                        /*parent=*/link.parent_ - 1,
                        /*rotParentToThis=*/joint_rot,
                        /*parentComToThisPivotOffset=*/joint_offset,
                        /*thisPivotToThisComOffset=*/com_offset);
                break;
            default:
                throw std::runtime_error("Unexpected joint type");
            }
        }

        wrapper.col_shapes_[i] = std::move(col_shape);
    }

    wrapper.multi_body_->finalizeMultiDof();
    world_->addMultiBody(wrapper.multi_body_.get());
    wrapper.multi_body_->setLinearDamping(0.0);
    wrapper.multi_body_->setAngularDamping(0.0);
    wrapper.multi_body_->clearForcesAndTorques();
    wrapper.multi_body_->clearVelocities();

    int dof_count = wrapper.multi_body_->getNumDofs();
    wrapper.joint_kp_.resize(dof_count);
    wrapper.joint_kd_.resize(dof_count);
    int dof_idx = 0;
    // The base link (index 0) has no actuated degrees of freedom
    for (std::size_t i = 1; i < robot->links_.size(); ++i) {
        // The first non-base link in Bullet has index 0
        const btMultibodyLink &link = wrapper.multi_body_->getLink(i - 1);
        for (int j = 0; j < link.m_dofCount; ++j) {
            wrapper.joint_kp_(dof_idx) = robot->links_[i].joint_kp_;
            wrapper.joint_kd_(dof_idx) = robot->links_[i].joint_kd_;
            ++dof_idx;
        }
    }
    wrapper.joint_target_pos_ = VectorX::Zero(dof_count);
    wrapper.joint_target_vel_ = VectorX::Zero(dof_count);
    wrapper.joint_motor_torques_ = VectorX::Zero(dof_count);

    // Add collision objects to world
    wrapper.colliders_.resize(wrapper.col_shapes_.size());
    for (std::size_t i = 0; i < wrapper.col_shapes_.size(); ++i) {
        auto collider = std::make_shared<btMultiBodyLinkCollider>(
                wrapper.multi_body_.get(), static_cast<int>(i) - 1);
        collider->setCollisionShape(wrapper.col_shapes_[i].get());
        collider->setFriction(robot->links_[i].friction_);
        collider->setUserPointer(const_cast<Robot *>(robot.get()));
        world_->addCollisionObject(collider.get(),
                                                             /*collisionFilterGroup=*/1,
                                                             /*collisionFilterMask=*/3);
        if (i == 0) {
            wrapper.multi_body_->setBaseCollider(collider.get());
        } else {
            wrapper.multi_body_->getLink(i - 1).m_collider = collider.get();
        }
        wrapper.colliders_[i] = std::move(collider);
    }

    // Initialize collision object world transforms
    btAlignedObjectArray<btQuaternion> scratch_q;
    btAlignedObjectArray<btVector3> scratch_m;
    wrapper.multi_body_->forwardKinematics(scratch_q, scratch_m);
    btAlignedObjectArray<btQuaternion> world_to_local;
    btAlignedObjectArray<btVector3> local_origin;
    wrapper.multi_body_->updateCollisionObjectWorldTransforms(world_to_local, local_origin);

    // Create joint motors
    wrapper.motors_.reserve(wrapper.multi_body_->getNumDofs());
    for (std::size_t i = 1; i < robot->links_.size(); ++i) {
        // The first non-base link in Bullet has index 0
        const btMultibodyLink &link = wrapper.multi_body_->getLink(i - 1);
        for (int dof_idx = 0; dof_idx < link.m_dofCount; ++dof_idx) {
            Scalar max_torque = robot->links_[i].joint_torque_;
            wrapper.motors_.push_back(std::make_shared<btMultiBodyJointMotor>(
                    /*body=*/wrapper.multi_body_.get(), /*link=*/i - 1,
                    /*linkDoF=*/dof_idx, /*desiredVelocity=*/0.0,
                    /*maxMotorImpulse=*/max_torque * time_step_));
            world_->addMultiBodyConstraint(wrapper.motors_.back().get());
        }
    }

    return robot_wrappers_.size() - 1;
}

Index BulletSimulation::addProp(std::shared_ptr<const Prop> prop, const Vector3 &pos, const Quaternion &rot) {
    prop_wrappers_.emplace_back(prop);
    BulletPropWrapper &wrapper = prop_wrappers_.back();

    Scalar mass = 0.0;
    switch (prop->shape_) {
    case PropShape::BOX:
        wrapper.col_shape_ = std::make_shared<btBoxShape>(
                bulletVector3FromEigen(prop->half_extents_));
        mass = 8 * prop->half_extents_.prod() * prop->density_;
        break;
    case PropShape::HEIGHTFIELD: {
        const HeightfieldProp &heightfield_prop =
                dynamic_cast<const HeightfieldProp &>(*prop);
        const MatrixX &heightfield = heightfield_prop.heightfield_;
        auto col_shape = std::make_shared<btHeightfieldTerrainShape>(
                heightfield.rows(), heightfield.cols(), heightfield.data(),
                /*heightScale=*/1.0, /*minHeight=*/0.0, /*maxHeight=*/1.0,
                /*upAxis=*/1, /*heightDataType=*/PHY_FLOAT, /*flipQuadEdges=*/false);
        Vector3 local_scaling =
                (2.0 * heightfield_prop.half_extents_).array() /
                Vector3(heightfield.rows() - 1, 1.0, heightfield.cols() - 1).array();
        col_shape->setLocalScaling(bulletVector3FromEigen(local_scaling));
        col_shape->buildAccelerator();
        wrapper.col_shape_ = std::move(col_shape);
        // Heightfields are always static (zero mass)
        break;
    }
    default:
        throw std::runtime_error("Unexpected prop shape");
    }

    btVector3 inertia(0, 0, 0);
    if (mass != 0) {
        wrapper.col_shape_->calculateLocalInertia(mass, inertia);
    }

    btRigidBody::btRigidBodyConstructionInfo rigid_body_info(
            /*mass=*/mass,
            /*motionState=*/nullptr,
            /*collisionShape=*/wrapper.col_shape_.get(),
            /*localInertia=*/inertia);
    rigid_body_info.m_friction = prop->friction_;
    wrapper.rigid_body_ = std::make_shared<btRigidBody>(rigid_body_info);
    wrapper.rigid_body_->setCenterOfMassTransform(btTransform(
            /*q=*/bulletQuaternionFromEigen(rot),
            /*c=*/bulletVector3FromEigen(pos)));
    wrapper.rigid_body_->setActivationState(DISABLE_DEACTIVATION);
    world_->addRigidBody(wrapper.rigid_body_.get(),
                                             /*collisionFilterGroup=*/2,
                                             /*collisionFilterMask=*/3);

    return prop_wrappers_.size() - 1;
}

void BulletSimulation::removeRobot(Index robot_idx) {
    auto it = robot_wrappers_.begin() + robot_idx;
    unregisterRobotWrapper(*it);
    robot_wrappers_.erase(it);
}

void BulletSimulation::removeProp(Index prop_idx) {
    auto it = prop_wrappers_.begin() + prop_idx;
    unregisterPropWrapper(*it);
    prop_wrappers_.erase(it);
}

std::shared_ptr<const Robot> BulletSimulation::getRobot(Index robot_idx) const {
    return robot_wrappers_[robot_idx].robot_;
}

std::shared_ptr<const Prop> BulletSimulation::getProp(Index prop_idx) const {
    return prop_wrappers_[prop_idx].prop_;
}

Index BulletSimulation::getRobotCount() const { return robot_wrappers_.size(); }

Index BulletSimulation::getPropCount() const { return prop_wrappers_.size(); }

Index BulletSimulation::findRobotIndex(const Robot &robot) const {
    for (std::size_t i = 0; i < robot_wrappers_.size(); ++i) {
        if (robot_wrappers_[i].robot_.get() == &robot) {
            return i;
        }
    }
    return -1;
}

Index BulletSimulation::findPropIndex(const Prop &prop) const {
    for (std::size_t i = 0; i < prop_wrappers_.size(); ++i) {
        if (prop_wrappers_[i].prop_.get() == &prop) {
            return i;
        }
    }
    return -1;
}

void BulletSimulation::unregisterRobotWrapper(BulletRobotWrapper &robot_wrapper) {
    for (auto motor : robot_wrapper.motors_) {
        world_->removeMultiBodyConstraint(motor.get());
    }
    for (auto collider : robot_wrapper.colliders_) {
        world_->removeCollisionObject(collider.get());
    }
    world_->removeMultiBody(robot_wrapper.multi_body_.get());
}

void BulletSimulation::unregisterPropWrapper(BulletPropWrapper &prop_wrapper) {
    world_->removeRigidBody(prop_wrapper.rigid_body_.get());
}

void BulletSimulation::getLinkTransform(Index robot_idx, Index link_idx, Ref<Matrix4> transform) const {
    btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    if (link_idx == 0) {
        // Base link
        transform = eigenMatrix4FromBullet(multi_body.getBaseWorldTransform());
    } else {
        transform = eigenMatrix4FromBullet(
                multi_body.getLink(link_idx - 1).m_cachedWorldTransform);
    }
}

void BulletSimulation::getPropTransform(Index prop_idx, Ref<Matrix4> transform) const {
    btRigidBody &rigid_body = *prop_wrappers_[prop_idx].rigid_body_;
    transform = eigenMatrix4FromBullet(rigid_body.getCenterOfMassTransform());
}

void BulletSimulation::getLinkVelocity(Index robot_idx, Index link_idx, Ref<Vector6> vel) const {
    btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    if (link_idx == 0) {
        // Base link
        vel.head<3>() = eigenVector3FromBullet(multi_body.getBaseOmega());
        vel.tail<3>() = eigenVector3FromBullet(multi_body.getBaseVel());
    } else {
        // TODO: implement for links other than the base
    }
}

Scalar BulletSimulation::getLinkMass(Index robot_idx, Index link_idx) const {
    btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    if (link_idx == 0) {
        return multi_body.getBaseMass();
    } else {
        return multi_body.getLinkMass(link_idx - 1);
    }
}

int BulletSimulation::getRobotDofCount(Index robot_idx) const {
    const btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    return multi_body.getNumDofs();
}

void BulletSimulation::getJointPositions(Index robot_idx, Ref<VectorX> pos) const {
    const btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    int offset = 0;
    for (int link_idx = 0; link_idx < multi_body.getNumLinks(); ++link_idx) {
        const btMultibodyLink &link = multi_body.getLink(link_idx);
        for (int pos_var_idx = 0; pos_var_idx < link.m_posVarCount; ++pos_var_idx) {
            pos(offset) = multi_body.getJointPosMultiDof(link_idx)[pos_var_idx];
            ++offset;
        }
    }
}

void BulletSimulation::getJointVelocities(Index robot_idx, Ref<VectorX> vel) const {
    const btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    int offset = 0;
    for (int link_idx = 0; link_idx < multi_body.getNumLinks(); ++link_idx) {
        const btMultibodyLink &link = multi_body.getLink(link_idx);
        for (int dof_idx = 0; dof_idx < link.m_dofCount; ++dof_idx) {
            vel(offset) = multi_body.getJointVelMultiDof(link_idx)[dof_idx];
            ++offset;
        }
    }
}

void BulletSimulation::getJointTargetPositions(Index robot_idx, Ref<VectorX> target_pos) const {
    target_pos = robot_wrappers_[robot_idx].joint_target_pos_;
}

void BulletSimulation::getJointTargetVelocities(Index robot_idx, Ref<VectorX> target_vel) const {
    target_vel = robot_wrappers_[robot_idx].joint_target_vel_;
}

void BulletSimulation::getJointMotorTorques(Index robot_idx, Ref<VectorX> motor_torques) const {
    motor_torques = robot_wrappers_[robot_idx].joint_motor_torques_;
}

void BulletSimulation::setJointTargets(Index robot_idx, const Ref<const VectorX> &target) {
    BulletRobotWrapper &wrapper = robot_wrappers_[robot_idx];
    const Robot *robot = wrapper.robot_.get();

    int dof_idx = 0;
    // The base link (index 0) has no actuated degrees of freedom
    for (std::size_t i = 1; i < robot->links_.size(); ++i) {
        // The first non-base link in Bullet has index 0
        const btMultibodyLink &link = wrapper.multi_body_->getLink(i - 1);
        for (int j = 0; j < link.m_dofCount; ++j) {
            Scalar joint_target;
            switch (robot->links_[i].joint_control_mode_) {
            case JointControlMode::POSITION:
                joint_target = target(dof_idx);
                wrapper.joint_target_pos_(dof_idx) = joint_target;
                wrapper.joint_target_vel_(dof_idx) = 0.0;
                break;
            case JointControlMode::VELOCITY:
                // TODO: allow changing this scaling factor
                joint_target = 5.0 * target(dof_idx);
                wrapper.joint_target_pos_(dof_idx) =
                        link.m_jointPos[j] + joint_target * time_step_;
                wrapper.joint_target_vel_(dof_idx) = joint_target;
                break;
            default:
                throw std::runtime_error("Unexpected joint control mode");
            }
            ++dof_idx;
        }
    }
}

void BulletSimulation::setJointTargetPositions(Index robot_idx, const Ref<const VectorX> &target_pos) {
    robot_wrappers_[robot_idx].joint_target_pos_ = target_pos;
}

void BulletSimulation::setJointTargetVelocities(Index robot_idx, const Ref<const VectorX> &target_vel) {
    robot_wrappers_[robot_idx].joint_target_vel_ = target_vel;
}

void BulletSimulation::addJointTorques(Index robot_idx, const Ref<const VectorX> &torque) {
    btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    assert(torque.size() == multi_body.getNumDofs());
    int offset = 0;
    for (int link_idx = 0; link_idx < multi_body.getNumLinks(); ++link_idx) {
        const btMultibodyLink &link = multi_body.getLink(link_idx);
        for (int dof = 0; dof < link.m_dofCount; ++dof) {
            multi_body.addJointTorqueMultiDof(link_idx, dof, torque(offset + dof));
        }
        offset += link.m_dofCount;
    }
}

void BulletSimulation::addLinkForceTorque(Index robot_idx, Index link_idx, const Ref<const Vector3> &force, const Ref<const Vector3> &torque) {
    btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    if (link_idx == 0) {
        multi_body.addBaseForce(bulletVector3FromEigen(force));
        multi_body.addBaseTorque(bulletVector3FromEigen(torque));
    } else {
        multi_body.addLinkForce(link_idx - 1, bulletVector3FromEigen(force));
        multi_body.addLinkTorque(link_idx - 1, bulletVector3FromEigen(torque));
    }
}

void BulletSimulation::getRobotWorldAABB(Index robot_idx, Ref<Vector3> lower, Ref<Vector3> upper) const {
    lower = Vector3::Constant(std::numeric_limits<Scalar>::infinity());
    upper = Vector3::Constant(-std::numeric_limits<Scalar>::infinity());
    const btMultiBody &multi_body = *robot_wrappers_[robot_idx].multi_body_;
    const btMultiBodyLinkCollider *base_collider = multi_body.getBaseCollider();
    btVector3 link_lower, link_upper;
    if (base_collider) {
        base_collider->getCollisionShape()->getAabb(
                multi_body.getBaseWorldTransform(), link_lower, link_upper);
        lower = lower.cwiseMin(eigenVector3FromBullet(link_lower));
        upper = upper.cwiseMax(eigenVector3FromBullet(link_upper));
    }
    for (int link_idx = 0; link_idx < multi_body.getNumLinks(); ++link_idx) {
        const btMultibodyLink &link = multi_body.getLink(link_idx);
        if (link.m_collider) {
            link.m_collider->getCollisionShape()->getAabb(link.m_cachedWorldTransform,
                                                                                                        link_lower, link_upper);
            lower = lower.cwiseMin(eigenVector3FromBullet(link_lower));
            upper = upper.cwiseMax(eigenVector3FromBullet(link_upper));
        }
    }
}

bool BulletSimulation::robotHasCollision(Index robot_idx) const {
    const Robot *robot = robot_wrappers_[robot_idx].robot_.get();
    int manifold_count = dispatcher_->getNumManifolds();
    for (int i = 0; i < manifold_count; ++i) {
        const btPersistentManifold *manifold =
                dispatcher_->getManifoldByIndexInternal(i);
        if (manifold->getBody0()->getUserPointer() == robot ||
                manifold->getBody1()->getUserPointer() == robot) {
            // Contact involves at least one of the robot's bodies
            int contact_count = manifold->getNumContacts();
            for (int j = 0; j < contact_count; ++j) {
                const btManifoldPoint &manifold_point = manifold->getContactPoint(j);
                if (manifold_point.getDistance() < 0) {
                    // Bodies are intersecting
                    return true;
                }
            }
        }
    }
    return false;
}

Scalar BulletSimulation::getTimeStep() const { return time_step_; }

Vector3 BulletSimulation::getGravity() const {
    return eigenVector3FromBullet(world_->getGravity());
}

void BulletSimulation::setGravity(const Ref<const Vector3> &gravity) {
    world_->setGravity(bulletVector3FromEigen(gravity));
}

void BulletSimulation::saveState() {
    auto serializer = std::make_shared<btDefaultSerializer>();
    int ser_flags = serializer->getSerializationFlags();
    serializer->setSerializationFlags(ser_flags | BT_SERIALIZE_CONTACT_MANIFOLDS);
    world_->serialize(serializer.get());

    auto bullet_file = std::make_shared<bParse::btBulletFile>(
            (char *)serializer->getBufferPointer(),
            serializer->getCurrentBufferSize());
    bullet_file->parse(false);
    if (bullet_file->ok()) {
        saved_state_ =
                BulletSavedState(std::move(serializer), std::move(bullet_file));
    } else {
        saved_state_ = BulletSavedState();
    }
}

void BulletSimulation::saveStateToFile(const char* filepath) {
    auto serializer = std::make_shared<btDefaultSerializer>();
    int ser_flags = serializer->getSerializationFlags();
    serializer->setSerializationFlags(ser_flags | BT_SERIALIZE_CONTACT_MANIFOLDS);
    world_->serialize(serializer.get());

    FILE* file = fopen(filepath, "wb");
	fwrite(serializer->getBufferPointer(), serializer->getCurrentBufferSize(), 1, file);
	fclose(file);
}

void BulletSimulation::restoreState() {
    if (saved_state_.bullet_file_ != nullptr) {
        auto importer = std::make_shared<btMultiBodyWorldImporter>(world_.get());
        importer->setImporterFlags(eRESTORE_EXISTING_OBJECTS);
        importer->convertAllObjects(saved_state_.bullet_file_.get());
    }
}

void BulletSimulation::restoreStateFromFile(const char* filepath) {
    auto importer = std::make_shared<btMultiBodyWorldImporter>(world_.get());
    importer->setImporterFlags(eRESTORE_EXISTING_OBJECTS);
    importer->loadFile(filepath);
}

void BulletSimulation::step() {
    // Run joint PD controllers
    for (Index robot_idx = 0; robot_idx < getRobotCount(); ++robot_idx) {
        BulletRobotWrapper &wrapper = robot_wrappers_[robot_idx];
        int dof_count = wrapper.multi_body_->getNumDofs();
        for (int dof_idx = 0; dof_idx < dof_count; ++dof_idx) {
            btMultiBodyJointMotor &motor = *wrapper.motors_[dof_idx];
            motor.setPositionTarget(wrapper.joint_target_pos_[dof_idx],
                                                            wrapper.joint_kp_[dof_idx]);
            motor.setVelocityTarget(wrapper.joint_target_vel_[dof_idx],
                                                            wrapper.joint_kd_[dof_idx]);
        }
    }
    world_->stepSimulation(time_step_, 0, time_step_);
    world_->forwardKinematics(); // Update m_cachedWorldTransform for every link
    // TODO: read back joint torques
}

bool BulletSimulation::OverlapFilterCallback::needBroadphaseCollision(
        btBroadphaseProxy *proxy0, btBroadphaseProxy *proxy1) const {
    // Ignore collisions between links rigidly attached to the same ancestor link
    auto *collider0 = btMultiBodyLinkCollider::upcast(
            static_cast<btCollisionObject *>(proxy0->m_clientObject));
    auto *collider1 = btMultiBodyLinkCollider::upcast(
            static_cast<btCollisionObject *>(proxy1->m_clientObject));
    if (collider0 && collider1 &&
            collider0->m_multiBody == collider1->m_multiBody) {
        // Both objects are links, and they are part of the same multibody
        btMultiBody *multi_body = collider0->m_multiBody;
        // Walk up the kinematic tree, traversing fixed joints only
        int link0 = collider0->m_link;
        while (link0 >= 0 &&
                     multi_body->getLink(link0).m_jointType == btMultibodyLink::eFixed) {
            link0 = multi_body->getParent(link0);
        }
        int link1 = collider1->m_link;
        while (link1 >= 0 &&
                     multi_body->getLink(link1).m_jointType == btMultibodyLink::eFixed) {
            link1 = multi_body->getParent(link1);
        }
        return link0 != link1;
    } else {
        return true;
    }
}

} // namespace robot_design
