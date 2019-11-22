#include <cstddef>
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
  for (std::size_t i = 0; i < layers_.size() - 1; ++i) {
    x = torch::tanh(layers_[i]->forward(x));
  }
  return layers_.back()->forward(x); // No activation after last layer
}

FCValueEstimator::FCValueEstimator(const Simulation &sim, Index robot_idx,
                                   const torch::Device &device, int batch_size,
                                   int epoch_count, int ensemble_size)
    : robot_idx_(robot_idx), device_(device), batch_size_(batch_size),
      epoch_count_(epoch_count) {
  dof_count_ = sim.getRobotDofCount(robot_idx);
  nets_.reserve(ensemble_size);
  optimizers_.reserve(ensemble_size);
  for (int k = 0; k < ensemble_size; ++k) {
    nets_.push_back(std::make_shared<FCValueNet>(getObservationSize(), 2, 64));
    nets_.back()->to(device);
    optimizers_.push_back(std::make_shared<torch::optim::Adam>(
        nets_.back()->parameters(), torch::optim::AdamOptions(1e-3)));
  }
}

int FCValueEstimator::getObservationSize() const {
  // Joint positions, joint velocities, base Y coordinate, base rotation matrix
  return 2 * dof_count_ + 1 + 9;
}

void FCValueEstimator::getObservation(const Simulation &sim,
                                      Ref<VectorX> obs) const {
  sim.getJointPositions(robot_idx_, obs.segment(0, dof_count_));
  sim.getJointVelocities(robot_idx_, obs.segment(dof_count_, dof_count_));
  Matrix4 base_transform;
  sim.getLinkTransform(robot_idx_, 0, base_transform);
  obs(2 * dof_count_) = base_transform(1, 3);
  Matrix3 base_rotation = base_transform.topLeftCorner<3, 3>();
  obs.segment(2 * dof_count_ + 1, 9) =
      Eigen::Map<VectorX>(base_rotation.data(), base_rotation.size());
}

void FCValueEstimator::estimateValue(const MatrixX &obs,
                                     Ref<VectorX> value_est) const {
  torch::Tensor obs_tensor = torchTensorFromEigenMatrix(obs);
  std::vector<torch::Tensor> ensemble_outputs;
  ensemble_outputs.reserve(nets_.size());
  for (std::size_t k = 0; k < nets_.size(); ++k) {
    ensemble_outputs.push_back(nets_[k]->forward(obs_tensor));
  }
  torch::Tensor ensemble_outputs_tensor = torch::stack(ensemble_outputs);
  torch::Tensor value_est_tensor =
      (torch::softmax(ensemble_outputs_tensor, 0) * ensemble_outputs_tensor)
          .sum(0);
  torchTensorToEigenVector(value_est_tensor, value_est);
}

void FCValueEstimator::train(const MatrixX &obs,
                             const Ref<const VectorX> &value) {
  assert(obs.cols() == value.size());
  size_t example_count = obs.cols();
  torch::data::samplers::RandomSampler sampler(0);
  for (int epoch_idx = 0; epoch_idx < epoch_count_; ++epoch_idx) {
    sampler.reset(example_count);
    auto index_batch = sampler.next(batch_size_);
    while (index_batch) {
      MatrixX obs_batch = obs(Eigen::all, *index_batch);
      VectorX value_batch = value(*index_batch);
      torch::Tensor obs_tensor = torchTensorFromEigenMatrix(obs_batch);
      torch::Tensor value_tensor = torchTensorFromEigenVector(value_batch);
      for (std::size_t k = 0; k < nets_.size(); ++k) {
        nets_[k]->zero_grad();
        torch::Tensor value_est_tensor =
            nets_[k]->forward(obs_tensor).flatten();
        torch::Tensor loss = torch::mse_loss(value_est_tensor, value_tensor);
        loss.backward();
        optimizers_[k]->step();
      }
      index_batch = sampler.next(batch_size_);
    }
  }
}

torch::Tensor FCValueEstimator::torchTensorFromEigenMatrix(
    const Ref<const MatrixX> &mat) const {
  // Create a row-major Torch tensor from a column-major Eigen matrix
  return torch::from_blob(const_cast<Scalar *>(mat.data()),
                          {mat.cols(), mat.rows()}, torch::dtype(SCALAR_DTYPE))
      .toType(TORCH_DTYPE)
      .to(device_);
}

torch::Tensor FCValueEstimator::torchTensorFromEigenVector(
    const Ref<const VectorX> &vec) const {
  return torch::from_blob(const_cast<Scalar *>(vec.data()), {vec.size()},
                          torch::dtype(SCALAR_DTYPE))
      .toType(TORCH_DTYPE)
      .to(device_);
}

void FCValueEstimator::torchTensorToEigenVector(const torch::Tensor &tensor,
                                                Ref<VectorX> vec) const {
  vec = Eigen::Map<VectorX>(tensor.cpu().toType(SCALAR_DTYPE).data<Scalar>(),
                            vec.size());
}

} // namespace robot_design
