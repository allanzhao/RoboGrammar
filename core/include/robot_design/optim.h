#pragma once

#include <ThreadPool.h>
#include <functional>
#include <robot_design/robot.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <robot_design/value.h>

namespace robot_design {

using MakeSimFunction = std::function<std::shared_ptr<Simulation>()>;

using ObjectiveFunction = std::function<Scalar(const Simulation &)>;

using Eigen::Ref;

class MPPIOptimizer {
public:
  MPPIOptimizer(Scalar kappa, Scalar discount_factor, int dof_count,
                int interval, int horizon, int sample_count, int thread_count,
                unsigned int seed, const MakeSimFunction &make_sim_fn,
                const ObjectiveFunction &objective_fn,
                const std::shared_ptr<const ValueEstimator> &value_estimator);
  void update();
  void advance(int step_count);

  MatrixX input_sequence_;

private:
  Scalar runSimulation(unsigned int sample_seed, int sample_idx);
  void advanceSimulation(int sample_idx, int step_count);
  void sampleInputSequence(Ref<MatrixX> rand_input_seq,
                           unsigned int sample_seed, int sample_idx) const;

  Scalar kappa_;
  Scalar discount_factor_;
  int dof_count_;
  int interval_;
  int horizon_;
  int sample_count_;
  unsigned int seed_;
  ObjectiveFunction objective_fn_;
  std::shared_ptr<const ValueEstimator> value_estimator_;
  std::vector<std::shared_ptr<Simulation>> sim_instances_;
  MatrixX final_obs_;
  ThreadPool thread_pool_;
};

struct SumOfSquaresObjective {
  Scalar operator()(const Simulation &sim) const;

  Vector6 base_vel_ref_ = Vector6::Zero();
  Vector6 base_vel_weight_ = Vector6::Zero();
  Scalar power_weight_ = 0.0;
};

struct DotProductObjective {
  Scalar operator()(const Simulation &sim) const;

  Vector3 base_dir_weight_ = Vector3::Zero();
  Vector3 base_up_weight_ = Vector3::Zero();
  Vector3 base_vel_weight_ = Vector3::Zero();
  Scalar power_weight_ = 0.0;
};

} // namespace robot_design
