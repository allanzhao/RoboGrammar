#include <robot_design/value.h>
#include <sstream>

namespace robot_design {

FCValueNet::FCValueNet(int obs_size, int hidden_layer_count,
                       int hidden_layer_size) {
  int last_layer_size = obs_size;
  for (int i = 0; i < hidden_layer_count; ++i) {
    std::ostringstream ss;
    ss << "fc" << i;
    layers_.push_back(register_module(
        ss.str(), torch::nn::Linear(last_layer_size, hidden_layer_size)));
    last_layer_size = hidden_layer_size;
  }
  std::ostringstream ss;
  ss << "fc" << hidden_layer_count;
  layers_.push_back(
      register_module(ss.str(), torch::nn::Linear(last_layer_size, 1)));
}

torch::Tensor FCValueNet::forward(torch::Tensor x) {
  for (int i = 0; i < layers_.size() - 1; ++i) {
    x = torch::relu(layers_[i]->forward(x));
  }
  return layers_.back()->forward(x);  // No activation after last layer
}

FCValueEstimator::FCValueEstimator(const Simulation &sim,
                                   const torch::Device &device)
    : device_(device) {
  net_ = std::make_shared<FCValueNet>(getObservationSize(sim), 2, 32);
  net_->to(device);
}

int FCValueEstimator::getObservationSize(const Simulation &sim) const {
  Index robot_idx = 0;
  return sim.getRobotDofCount(robot_idx) * 2;  // Joint positions and velocities
}

void FCValueEstimator::getObservation(const Simulation &sim,
                                      Eigen::Ref<VectorX> obs) const {
  Index robot_idx = 0;
  int dof_count = sim.getRobotDofCount(robot_idx);
  sim.getJointPositions(robot_idx, obs.head(dof_count));
  sim.getJointVelocities(robot_idx, obs.tail(dof_count));
}

void FCValueEstimator::estimateValue(const MatrixX &obs,
                                     Eigen::Ref<VectorX> value) const {
  // Create a row-major Torch tensor from a column-major Eigen matrix
  torch::Tensor obs_tensor =
      torch::from_blob(const_cast<Scalar *>(obs.data()),
                       {obs.cols(), obs.rows()}, torch::dtype(SCALAR_DTYPE))
          .toType(TORCH_DTYPE)
          .to(device_);
  torch::Tensor value_tensor = net_->forward(obs_tensor).cpu()
      .toType(SCALAR_DTYPE);
  value = Eigen::Map<VectorX>(value_tensor.data<Scalar>(), value.size());
}

}  // namespace robot_design
