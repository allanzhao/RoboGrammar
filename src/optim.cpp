#include <random>
#include <robot_design/optim.h>

namespace robot_design {

MPPIOptimizer::MPPIOptimizer(
    Scalar kappa, Scalar discount_factor, int dof_count, int interval,
    int horizon, int sample_count, int thread_count, unsigned int seed,
    const MakeSimFunction &make_sim_fn, const ObjectiveFunction &objective_fn,
    const std::shared_ptr<const FCValueEstimator> &value_estimator)
    : kappa_(kappa), discount_factor_(discount_factor), dof_count_(dof_count),
      interval_(interval), horizon_(horizon), sample_count_(sample_count),
      seed_(seed), objective_fn_(objective_fn),
      value_estimator_(value_estimator), next_thread_id_(0),
      thread_pool_(thread_count) {
  // Create a separate simulation instance for each thread
  sim_instances_.reserve(thread_count);
  for (int i = 0; i < thread_count; ++i) {
    sim_instances_.push_back(std::move(make_sim_fn()));
  }

  input_sequence_ = MatrixX::Zero(dof_count, horizon);
  final_obs_.resize(value_estimator->getObservationSize(), sample_count);
}

void MPPIOptimizer::update() {
  // Sample trajectories with different seeds
  std::vector<std::future<double>> sim_results;
  sim_results.reserve(sample_count_);
  for (int k = 0; k < sample_count_; ++k) {
    sim_results.emplace_back(thread_pool_.enqueue(
        &MPPIOptimizer::runSimulation, this, k, seed_ + k));
  }

  // Wait on results
  VectorX sim_rewards(sample_count_);
  for (int k = 0; k < sample_count_; ++k) {
    sim_rewards(k) = sim_results[k].get();
  }

  // Estimate (discounted) values of final states and add to simulation reward
  VectorX final_value_est(sample_count_);
  value_estimator_->estimateValue(final_obs_, final_value_est);
  sim_rewards += final_value_est * std::pow(discount_factor_, horizon_);

  MatrixX input_sequence_sum = MatrixX::Zero(dof_count_, horizon_);
  Scalar seq_weight_sum = 0.0;
  MatrixX rand_input_seq;
  Scalar max_reward = sim_rewards.maxCoeff();
  for (int k = 0; k < sample_count_; ++k) {
    // Recreate the same input sequence used for the simulation
    sampleInputSequence(rand_input_seq, seed_ + k);
    Scalar seq_weight = std::exp(kappa_ * (sim_rewards(k) - max_reward));
    input_sequence_sum += rand_input_seq * seq_weight;
    seq_weight_sum += seq_weight;
  }
  input_sequence_ = input_sequence_sum / seq_weight_sum;

  seed_ += sample_count_;
}

void MPPIOptimizer::advance(int step_count) {
  Index robot_idx = 0;  // TODO: don't assume there is only one robot
  for (auto &sim : sim_instances_) {
    for (int j = 0; j < step_count; ++j) {
      for (int i = 0; i < interval_; ++i) {
        sim->setJointTargetPositions(robot_idx, input_sequence_.col(j));
        sim->step();
      }
    }
  }
  input_sequence_.leftCols(horizon_ - step_count) = input_sequence_.rightCols(horizon_ - step_count);
  input_sequence_.rightCols(step_count) = MatrixX::Zero(dof_count_, step_count);
}

Scalar MPPIOptimizer::runSimulation(int sample_idx, unsigned int sample_seed) {
  thread_local int thread_id = next_thread_id_++;
  Simulation &sim = *sim_instances_[thread_id];
  Index robot_idx = 0;  // TODO: don't assume there is only one robot
  MatrixX rand_input_seq;
  sampleInputSequence(rand_input_seq, sample_seed);
  sim.saveState();
  Scalar reward = 0.0;
  Scalar discount_prod = 1.0;
  for (int j = 0; j < horizon_; ++j) {
    for (int i = 0; i < interval_; ++i) {
      sim.setJointTargetPositions(robot_idx, rand_input_seq.col(j));
      sim.step();
      reward += objective_fn_(sim) * discount_prod;
    }
    discount_prod *= discount_factor_;
  }
  // Collect observation for final state
  value_estimator_->getObservation(sim, final_obs_.col(sample_idx));
  sim.restoreState();
  return reward;
}

void MPPIOptimizer::sampleInputSequence(MatrixX &rand_input_seq, unsigned int sample_seed) const {
  std::mt19937 generator(sample_seed);
  std::normal_distribution<Scalar> distribution(0.0, 0.2);
  rand_input_seq = input_sequence_ + MatrixX::NullaryExpr(dof_count_, horizon_,
      [&]() { return distribution(generator); });
}

}  // namespace robot_design
