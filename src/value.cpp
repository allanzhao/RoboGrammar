#include <robot_design/value.h>
#include <sstream>

namespace robot_design {

FCValueNet::FCValueNet(int obs_size, int hidden_layer_count,
                       int hidden_layer_size) {
  int last_layer_size = obs_size;
  for (int i = 0; i < hidden_layer_count; ++i) {
    std::ostringstream ss;
    ss << "fc" << i;
    layers_.push_back(register_module(ss.str(),
        torch::nn::Linear(last_layer_size, hidden_layer_size)));
    last_layer_size = hidden_layer_size;
  }
  std::ostringstream ss;
  ss << "fc" << hidden_layer_count;
  layers_.push_back(register_module(ss.str(),
      torch::nn::Linear(last_layer_size, 1)));
}

torch::Tensor FCValueNet::forward(torch::Tensor x) {
  for (int i = 0; i < layers_.size() - 1; ++i) {
    x = torch::relu(layers_[i]->forward(x));
  }
  return layers_.back()->forward(x);  // No activation after last layer
}

int FCValueEstimator::getObservationSize(const Simulation &sim) const {
  Index robot_idx = 0;
  return sim.getRobotDofCount(robot_idx) * 2;  // Joint positions and velocities
}

void FCValueEstimator::getObservation(const Simulation &sim, Ref<VectorX> obs) const {
  Index robot_idx = 0;
  int dof_count = sim.getRobotDofCount(robot_idx);
  sim.getJointPositions(robot_idx, obs.head(dof_count));
  sim.getJointVelocities(robot_idx, obs.tail(dof_count));
}

void FCValueEstimator::estimateValue(const MatrixX &obs, Ref<VectorX> value) const {
  // TODO
}

}  // namespace robot_design
