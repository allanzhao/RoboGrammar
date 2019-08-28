#pragma once

#include <functional>
#include <robot_design/robot.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <robot_design/value.h>
#include <ThreadPool.h>

namespace robot_design {

using MakeSimFunction = std::function<std::shared_ptr<Simulation>()>;

using ObjectiveFunction = std::function<Scalar(const Simulation &)>;

class MPPIOptimizer {
public:
  MPPIOptimizer(
      Scalar kappa, Scalar discount_factor, int dof_count, int interval,
      int horizon, int sample_count, int thread_count, unsigned int seed,
      const MakeSimFunction &make_sim_fn, const ObjectiveFunction &objective_fn,
      const std::shared_ptr<const FCValueEstimator> &value_estimator);
  void update();
  void advance(int step_count);

  MatrixX input_sequence_;

private:
  Scalar runSimulation(int sample_idx, unsigned int sample_seed);
  void advanceSimulation(int sample_idx, int step_count);
  void sampleInputSequence(MatrixX &rand_input_seq, unsigned int sample_seed) const;

  Scalar kappa_;
  Scalar discount_factor_;
  int dof_count_;
  int interval_;
  int horizon_;
  int sample_count_;
  unsigned int seed_;
  ObjectiveFunction objective_fn_;
  std::shared_ptr<const FCValueEstimator> value_estimator_;
  std::vector<std::shared_ptr<Simulation>> sim_instances_;
  MatrixX final_obs_;
  ThreadPool thread_pool_;
};

}  // namespace robot_design
