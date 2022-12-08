#include <random>
#include <robot_design/optim.h>

namespace robot_design {

DefaultInputSampler::DefaultInputSampler(Scalar history_std_dev,
                                         Scalar warm_start_std_dev)
    : history_std_dev_(history_std_dev),
      warm_start_std_dev_(warm_start_std_dev) {}

void DefaultInputSampler::sampleInputSequence(
    Ref<MatrixX> input_seq, unsigned int sample_seed, int sample_idx,
    const Ref<const MatrixX> &last_input_seq,
    const Ref<const MatrixX> &history) const {
  int dof_count = last_input_seq.rows();
  int horizon = last_input_seq.cols();
  int history_len = history.cols();

  Scalar std_dev;
  // Half of the samples are based on repeating past motion ("history")
  if (history_len >= horizon && sample_idx % 2 == 0) {
    // The repetition period should range from (horizon / 2 + 1) to horizon
    int repeat_len = (sample_idx / 2) % (horizon / 2) + horizon / 2 + 1;
    for (int j = 0; j < horizon; ++j) {
      // Linearly interpolate to avoid sudden changes in the inputs
      Scalar t = static_cast<Scalar>(j + 1) / horizon;
      input_seq.col(j) =
          t * history.rightCols(repeat_len).replicate(1, horizon).col(j) +
          (1.0 - t) * last_input_seq.col(j);
    }
    std_dev = history_std_dev_;
  } else {
    input_seq = last_input_seq;
    std_dev = warm_start_std_dev_;
  }
  std::mt19937 generator(sample_seed + sample_idx);
  std::normal_distribution<Scalar> distribution(0.0, std_dev);
  Eigen::Vector<Scalar, 5> filter_coeffs;
  // FIR filter with passband below 2 Hz, stopband above 4 Hz at f_s = 15 Hz
  filter_coeffs << 0.10422766377112629, 0.3239870556899027, 0.3658903830367387,
      0.3239870556899027, 0.10422766377112629;
  int filter_len = filter_coeffs.size();
  MatrixX noise =
      MatrixX::NullaryExpr(dof_count, horizon + filter_len - 1,
                           [&]() { return distribution(generator); });
  for (int j = 0; j < horizon; ++j) {
    input_seq.col(j) +=
        noise.block(0, j, dof_count, filter_len) * filter_coeffs;
  }
}

void ConstantInputSampler::sampleInputSequence(
    Ref<MatrixX> input_seq, unsigned int sample_seed, int sample_idx,
    const Ref<const MatrixX> &last_input_seq,
    const Ref<const MatrixX> &history) const {
  input_seq = samples_.block(sample_idx * input_seq.rows(), 0, input_seq.rows(),
                             input_seq.cols());
}

MPPIOptimizer::MPPIOptimizer(
    Scalar kappa, Scalar discount_factor, int dof_count, int interval,
    int horizon, int sample_count, int thread_count, unsigned int seed,
    const MakeSimFunction &make_sim_fn, const ObjectiveFunction &objective_fn,
    const std::shared_ptr<const ValueEstimator> &value_estimator,
    const std::shared_ptr<const InputSampler> &input_sampler)
    : kappa_(kappa), discount_factor_(discount_factor), dof_count_(dof_count),
      interval_(interval), horizon_(horizon), sample_count_(sample_count),
      seed_(seed), objective_fn_(objective_fn),
      value_estimator_(value_estimator), input_sampler_(input_sampler),
      thread_pool_(thread_count) {
  // Create a separate simulation instance for each sample
  sim_instances_.reserve(sample_count);
  for (int i = 0; i < sample_count; ++i) {
    sim_instances_.push_back(make_sim_fn());
  }

  input_sequence_ = MatrixX::Zero(dof_count, horizon);
  final_obs_.resize(value_estimator->getObservationSize(), sample_count);
  history_ = MatrixX::Zero(dof_count, horizon);
  total_step_count_ = 0;
}

void MPPIOptimizer::update() {
  // Sample trajectories with different seeds
  std::vector<std::future<double>> sim_results;
  sim_results.reserve(sample_count_);
  for (int k = 0; k < sample_count_; ++k) {
    sim_results.emplace_back(
        thread_pool_.enqueue(&MPPIOptimizer::runSimulation, this, seed_, k));
  }

  // Wait on results
  VectorX sim_returns(sample_count_);
  for (int k = 0; k < sample_count_; ++k) {
    sim_returns(k) = sim_results[k].get();
  }

  // Estimate (discounted) values of final states and add to simulation return
  VectorX final_value_est(sample_count_);
  value_estimator_->estimateValue(final_obs_, final_value_est);
  sim_returns += final_value_est * std::pow(discount_factor_, horizon_);

  MatrixX input_sequence_sum = MatrixX::Zero(dof_count_, horizon_);
  Scalar seq_weight_sum = 0.0;
  MatrixX rand_input_seq(dof_count_, horizon_);
  Scalar max_return = sim_returns.maxCoeff();
  for (int k = 0; k < sample_count_; ++k) {
    // Recreate the same input sequence used for the simulation
    input_sampler_->sampleInputSequence(rand_input_seq, seed_, k,
                                        input_sequence_, history_);
    Scalar seq_weight = std::exp(kappa_ * (sim_returns(k) - max_return));
    input_sequence_sum += rand_input_seq * seq_weight;
    seq_weight_sum += seq_weight;
  }
  input_sequence_ = input_sequence_sum / seq_weight_sum;

  seed_ += sample_count_;
}

void MPPIOptimizer::advance(int step_count) {
  std::vector<std::future<void>> futures;
  futures.reserve(sample_count_);
  for (int k = 0; k < sample_count_; ++k) {
    futures.emplace_back(thread_pool_.enqueue(&MPPIOptimizer::advanceSimulation,
                                              this, k, step_count));
  }

  // Wait for all simulation instances to advance
  for (int k = 0; k < sample_count_; ++k) {
    futures[k].get();
  }
  // Shift the contents of history_ to the left by step_count
  for (int j = 0; j < history_.cols() - step_count; ++j) {
    history_.col(j) = history_.col(j + step_count);
  }
  // Copy the inputs applied to the simulation into history_
  history_.rightCols(step_count) = input_sequence_.leftCols(step_count);
  // Shift the contents of input_sequence_ to the left by step_count
  for (int j = 0; j < horizon_ - step_count; ++j) {
    input_sequence_.col(j) = input_sequence_.col(j + step_count);
  }
  // Fill in the remaining values with zeroes
  input_sequence_.rightCols(step_count) = MatrixX::Zero(dof_count_, step_count);

  total_step_count_ += step_count;
}

int MPPIOptimizer::getSampleCount() const { return sample_count_; }

void MPPIOptimizer::setSampleCount(int sample_count) {
  if (sample_count <= sample_count_) {
    sample_count_ = sample_count;
    sim_instances_.resize(sample_count);
    final_obs_.resize(value_estimator_->getObservationSize(), sample_count);
  } else {
    // Increasing the sample count is not supported yet
    throw std::invalid_argument("Cannot increase sample count");
  }
}

Scalar MPPIOptimizer::runSimulation(unsigned int sample_seed, int sample_idx) {
  Simulation &sim = *sim_instances_[sample_idx];
  Index robot_idx = 0; // TODO: don't assume there is only one robot
  MatrixX rand_input_seq(dof_count_, horizon_);
  input_sampler_->sampleInputSequence(rand_input_seq, sample_seed, sample_idx,
                                      input_sequence_, history_);
  sim.saveState();
  Scalar sim_return = 0.0;
  Scalar discount_prod = 1.0;
  for (int j = 0; j < horizon_; ++j) {
    for (int i = 0; i < interval_; ++i) {
      sim.setJointTargets(robot_idx, rand_input_seq.col(j));
      sim.step();
      sim_return += objective_fn_(sim) * discount_prod;
    }
    discount_prod *= discount_factor_;
  }
  // Collect observation for final state
  value_estimator_->getObservation(sim, final_obs_.col(sample_idx));
  sim.restoreState();
  return sim_return;
}

void MPPIOptimizer::advanceSimulation(int sample_idx, int step_count) {
  Simulation &sim = *sim_instances_[sample_idx];
  Index robot_idx = 0; // TODO: don't assume there is only one robot
  for (int j = 0; j < step_count; ++j) {
    for (int i = 0; i < interval_; ++i) {
      sim.setJointTargets(robot_idx, input_sequence_.col(j));
      sim.step();
    }
  }
}

Scalar SumOfSquaresObjective::operator()(const Simulation &sim) const {
  Scalar cost = 0.0;
  for (Index robot_idx = 0; robot_idx < sim.getRobotCount(); ++robot_idx) {
    int dof_count = sim.getRobotDofCount(robot_idx);
    Vector6 base_vel;
    sim.getLinkVelocity(robot_idx, 0, base_vel);
    VectorX joint_vel(dof_count);
    sim.getJointVelocities(robot_idx, joint_vel);
    VectorX motor_torques(dof_count);
    sim.getJointMotorTorques(robot_idx, motor_torques);
    // Velocity error term
    Vector6 vel_error = base_vel - base_vel_ref_;
    cost += vel_error.transpose() * base_vel_weight_.asDiagonal() * vel_error;
    // Power consumption term
    Scalar power = motor_torques.dot(joint_vel);
    cost += power_weight_ * power * power;
  }
  // Negate the cost to turn it into a reward
  return -cost;
}

Scalar DotProductObjective::operator()(const Simulation &sim) const {
  Scalar reward = 0.0;
  for (Index robot_idx = 0; robot_idx < sim.getRobotCount(); ++robot_idx) {
    int dof_count = sim.getRobotDofCount(robot_idx);
    Matrix4 base_transform;
    sim.getLinkTransform(robot_idx, 0, base_transform);
    Vector6 base_vel;
    sim.getLinkVelocity(robot_idx, 0, base_vel);
    VectorX joint_vel(dof_count);
    sim.getJointVelocities(robot_idx, joint_vel);
    VectorX motor_torques(dof_count);
    sim.getJointMotorTorques(robot_idx, motor_torques);
    // Base direction term
    Vector3 base_dir = base_transform.block<3, 1>(0, 0);
    reward += base_dir.dot(base_dir_weight_);
    // Base up vector term
    Vector3 base_up = base_transform.block<3, 1>(0, 1);
    reward += base_up.dot(base_up_weight_);
    reward *= 0.3; // change in reward carlo	
    // Base velocity term
    reward += base_vel.tail<3>().dot(base_vel_weight_);
    // Power consumption term
    Scalar power = motor_torques.dot(joint_vel);
    reward += power_weight_ * power;
  }
  return reward;
}

} // namespace robot_design
