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
                                   const torch::Device &device,
                                   int batch_size, int epoch_count)
    : device_(device), batch_size_(batch_size), epoch_count_(epoch_count) {
  net_ = std::make_shared<FCValueNet>(getObservationSize(sim), 2, 32);
  net_->to(device);
  optimizer_ = std::make_shared<torch::optim::Adam>(
      net_->parameters(), torch::optim::AdamOptions(1e-3));
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
                                     Eigen::Ref<VectorX> value_est) const {
  torch::Tensor obs_tensor = torchTensorFromEigenMatrix(obs);
  torch::Tensor value_est_tensor = net_->forward(obs_tensor);
  torchTensorToEigenVector(value_est_tensor, value_est);
}

void FCValueEstimator::train(const MatrixX &obs,
                             const Eigen::Ref<const VectorX> &value) {
  assert(obs.cols() == value.size());
  size_t example_count = obs.cols();
  torch::data::samplers::RandomSampler sampler(0);
  for (int epoch_idx = 0; epoch_idx < epoch_count_; ++epoch_idx) {
    sampler.reset(example_count);
    auto index_batch = sampler.next(batch_size_);
    while (index_batch) {
      net_->zero_grad();
      MatrixX obs_batch = obs(Eigen::all, *index_batch);
      VectorX value_batch = value(*index_batch);
      torch::Tensor obs_tensor = torchTensorFromEigenMatrix(obs_batch);
      torch::Tensor value_tensor = torchTensorFromEigenVector(value_batch);
      torch::Tensor value_est_tensor = net_->forward(obs_tensor).flatten();
      torch::Tensor loss = torch::mse_loss(value_est_tensor, value_tensor);
      loss.backward();
      optimizer_->step();
      index_batch = sampler.next(batch_size_);
    }
  }
}

torch::Tensor FCValueEstimator::torchTensorFromEigenMatrix(
    const MatrixX &mat) const {
  // Create a row-major Torch tensor from a column-major Eigen matrix
  return torch::from_blob(const_cast<Scalar *>(mat.data()),
                          {mat.cols(), mat.rows()}, torch::dtype(SCALAR_DTYPE))
      .toType(TORCH_DTYPE)
      .to(device_);
}

torch::Tensor FCValueEstimator::torchTensorFromEigenVector(
    const Eigen::Ref<const VectorX> &vec) const {
  return torch::from_blob(const_cast<Scalar *>(vec.data()),
                          {vec.size()}, torch::dtype(SCALAR_DTYPE))
      .toType(TORCH_DTYPE)
      .to(device_);
}

void FCValueEstimator::torchTensorToEigenVector(const torch::Tensor &tensor,
                                                Eigen::Ref<VectorX> vec) const {
  vec = Eigen::Map<VectorX>(tensor.cpu().toType(SCALAR_DTYPE).data<Scalar>(),
                            vec.size());
}

}  // namespace robot_design
