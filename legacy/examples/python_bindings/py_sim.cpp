#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <robot_design/sim.h>

namespace py = pybind11;
namespace rd = robot_design;

void initSim(py::module &m) {
  py::class_<rd::Simulation, std::shared_ptr<rd::Simulation>>(m, "Simulation")
      .def("add_robot", &rd::Simulation::addRobot)
      .def("add_prop", &rd::Simulation::addProp)
      .def("remove_robot", &rd::Simulation::removeRobot)
      .def("remove_prop", &rd::Simulation::removeProp)
      .def("get_robot", &rd::Simulation::getRobot)
      .def("get_prop", &rd::Simulation::getProp)
      .def("get_robot_count", &rd::Simulation::getRobotCount)
      .def("get_prop_count", &rd::Simulation::getPropCount)
      .def("find_robot_index", &rd::Simulation::findRobotIndex)
      .def("find_prop_index", &rd::Simulation::findPropIndex)
      .def("get_link_transform", &rd::Simulation::getLinkTransform)
      .def("get_prop_transform", &rd::Simulation::getPropTransform)
      .def("get_link_velocity", &rd::Simulation::getLinkVelocity)
      .def("get_link_mass", &rd::Simulation::getLinkMass)
      .def("get_robot_dof_count", &rd::Simulation::getRobotDofCount)
      .def("get_joint_positions", &rd::Simulation::getJointPositions)
      .def("get_joint_velocities", &rd::Simulation::getJointVelocities)
      .def("get_joint_target_positions",
           &rd::Simulation::getJointTargetPositions)
      .def("get_joint_target_velocities",
           &rd::Simulation::getJointTargetVelocities)
      .def("get_joint_motor_torques", &rd::Simulation::getJointMotorTorques)
      .def("set_joint_targets", &rd::Simulation::setJointTargets)
      .def("set_joint_target_positions",
           &rd::Simulation::setJointTargetPositions)
      .def("set_joint_target_velocities",
           &rd::Simulation::setJointTargetVelocities)
      .def("add_joint_torques", &rd::Simulation::addJointTorques)
      .def("add_link_force_torque", &rd::Simulation::addLinkForceTorque)
      .def("get_robot_world_aabb", &rd::Simulation::getRobotWorldAABB)
      .def("robot_has_collision", &rd::Simulation::robotHasCollision)
      .def("get_time_step", &rd::Simulation::getTimeStep)
      .def("get_gravity", &rd::Simulation::getGravity)
      .def("set_gravity", &rd::Simulation::setGravity)
      .def("save_state", &rd::Simulation::saveState)
      .def("save_state_to_file", &rd::Simulation::saveStateToFile)
      .def("restore_state", &rd::Simulation::restoreState)
      .def("restore_state_from_file", &rd::Simulation::restoreStateFromFile)
      .def("step", &rd::Simulation::step);

  py::class_<rd::BulletSimulation, rd::Simulation, 
             std::shared_ptr<rd::BulletSimulation>>(m, "BulletSimulation")
      .def(py::init<rd::Scalar>())
      .def(py::init<>());
}
