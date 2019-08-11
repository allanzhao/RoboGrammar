#include <iostream>
#include <robot_design/control.h>
#include <robot_design/utils.h>

namespace robot_design {

PDController::PDController(const Robot &robot, Simulation &sim)
    : robot_(robot), sim_(sim) {
  Index robot_idx = sim_.findRobotIndex(robot);
  int dof_count = sim_.getRobotDofCount(robot_idx);
  kp_ = VectorX::Zero(dof_count);
  kd_ = VectorX::Zero(dof_count);
  pos_target_ = VectorX::Zero(dof_count);
  vel_target_ = VectorX::Zero(dof_count);
}

void PDController::update() {
  Index robot_idx = sim_.findRobotIndex(robot_);
  sim_.getJointPositions(robot_idx, pos_);
  sim_.getJointVelocities(robot_idx, vel_);
  torque_ = kp_.cwiseProduct(pos_target_ - pos_) +
            kd_.cwiseProduct(vel_target_ - vel_);
  sim_.addJointTorques(robot_idx, torque_);
}

MPCController::MPCController(const Robot &robot, Simulation &sim, int horizon,
                             int interval, const MakeSimFunction &make_sim_fn,
                             const ObjectiveFunction &objective_fn,
                             int thread_count)
    : robot_(robot), sim_(sim), horizon_(horizon), interval_(interval),
      objective_fn_(objective_fn), thread_pool_(thread_count), step_count_(0),
      dx_(1e-8) {
  Index robot_idx = sim_.findRobotIndex(robot);
  int dof_count = sim_.getRobotDofCount(robot_idx);

  // Create independent simulation instances for finite differencing
  int instance_count = 2 * dof_count * horizon;
  sim_instances_.reserve(instance_count);
  for (int i = 0; i < instance_count; ++i) {
    sim_instances_.push_back(make_sim_fn());
  }
  sim_results_.resize(instance_count);

  // Define initial input trajectory
  // Trajectory contains (horizon + 1) steps because the first step is fixed
  input_trajectory_ = MatrixX::Zero(dof_count, horizon + 1);
}

void MPCController::update() {
  Index robot_idx = sim_.findRobotIndex(robot_);
  int dof_count = sim_.getRobotDofCount(robot_idx);

  // Apply the best inputs found so far for this time step
  VectorX inputs = input_trajectory_.col(0);
  sim_.addJointTorques(robot_idx, inputs);

  if (step_count_ % interval_ == 0) {
    if (step_count_ > 0) {
      // Update input trajectory using results from previous interval
      MatrixX objective_grad = MatrixX::Zero(dof_count, horizon_);
      for (int j = 0; j < horizon_; ++j) {
        for (int i = 0; i < dof_count; ++i) {
          Scalar fp = sim_results_[2 * (dof_count * j + i)].get();
          Scalar fm = sim_results_[2 * (dof_count * j + i) + 1].get();
          objective_grad(i, j) = clamp((fp - fm) / (2 * dx_), -1.0, 1.0);
        }
      }
      input_trajectory_.rightCols(horizon_) += objective_grad;
      input_trajectory_.leftCols(horizon_) = input_trajectory_.rightCols(horizon_);
    }

    for (int i = 0; i < sim_instances_.size(); ++i) {
      sim_results_[i] = thread_pool_.enqueue(&MPCController::runSimulation, this, i);
    }
  }

  ++step_count_;
}

void MPCController::perturbInputs(MatrixX &input_trajectory, int sim_idx) const {
  Scalar amount = (sim_idx % 2 == 0) ? dx_ : -dx_;
  input_trajectory.reshaped()(sim_idx / 2) += amount;
}

Scalar MPCController::runSimulation(int sim_idx) {
  Simulation &sim_instance = *sim_instances_[sim_idx];
  Index robot_idx = sim_instance.findRobotIndex(robot_);

  // "Catch up" with the main simulation by applying the same inputs
  VectorX inputs = input_trajectory_.col(0);
  for (int j = 0; j < interval_; ++j) {
    sim_instance.addJointTorques(robot_idx, inputs);
    sim_instance.step();
  }

  sim_instance.saveState();

  // Apply the rest of the inputs plus a perturbation
  MatrixX future_inputs = input_trajectory_.rightCols(horizon_);
  perturbInputs(future_inputs, sim_idx);
  Scalar objective_value = 0.0;
  for (int j = 0; j < horizon_; ++j) {
    VectorX inputs = future_inputs.col(j);
    for (int i = 0; i < interval_; ++i) {
      sim_instance.addJointTorques(robot_idx, inputs);
      sim_instance.step();
      objective_value += objective_fn_(sim_instance);
    }
  }

  sim_instance.restoreState();
  return objective_value;
}

}  // namespace robot_design
