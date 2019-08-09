#pragma once

#include <functional>
#include <robot_design/robot.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <ThreadPool.h>

namespace robot_design {

class PDController {
public:
  PDController(const Robot &robot, Simulation &sim);
  void update();

  VectorX kp_;
  VectorX kd_;
  VectorX pos_target_;
  VectorX vel_target_;

private:
  const Robot &robot_;
  Simulation &sim_;
  VectorX pos_;
  VectorX vel_;
  VectorX torque_;
};

using MakeSimFunction = std::function<std::shared_ptr<Simulation>()>;

using ObjectiveFunction = std::function<Scalar(const Simulation &)>;

class MPCController {
public:
  MPCController(const Robot &robot, Simulation &sim, int horizon, int period,
                const MakeSimFunction &make_sim_fn,
                const ObjectiveFunction &objective_fn, int thread_count);
  void update();

private:
  Scalar runSimulation(int sim_index);

  const Robot &robot_;
  Simulation &sim_;
  int horizon_;
  int period_;
  ObjectiveFunction objective_fn_;
  ThreadPool thread_pool_;
  std::vector<std::shared_ptr<Simulation>> sim_instances_;
  std::vector<std::future<Scalar>> sim_results_;
  MatrixX inputs_;
};

}  // namespace robot_design
