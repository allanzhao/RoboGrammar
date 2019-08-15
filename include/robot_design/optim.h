#pragma once

#include <atomic>
#include <functional>
#include <robot_design/robot.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <ThreadPool.h>

namespace robot_design {

using MakeSimFunction = std::function<std::shared_ptr<Simulation>()>;

using ObjectiveFunction = std::function<Scalar(const Simulation &, int)>;

class MPPIOptimizer {
public:
  MPPIOptimizer(Scalar kappa, int dof_count, int horizon, int sample_count,
                int thread_count, unsigned int seed,
                const MakeSimFunction &make_sim_fn,
                const ObjectiveFunction &objective_fn);
  void update();

  MatrixX input_sequence_;

private:
  Scalar runSimulation(int sample_seed);
  void sampleInputSequence(MatrixX &rand_input_seq, int sample_seed) const;

  Scalar kappa_;
  int dof_count_;
  int horizon_;
  int sample_count_;
  unsigned int seed_;
  ObjectiveFunction objective_fn_;
  std::vector<std::shared_ptr<Simulation>> sim_instances_;
  std::atomic<int> next_thread_id_;
  ThreadPool thread_pool_;
};

}  // namespace robot_design
