#pragma once

#include <memory>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <torch/torch.h>
#include <type_traits>
#include <vector>

namespace robot_design {

// Torch dtype corresponding to Scalar
constexpr torch::Dtype SCALAR_DTYPE =
    std::is_same<Scalar, double>::value ? torch::kFloat64 : torch::kFloat32;

// Torch dtype used internally
constexpr torch::Dtype TORCH_DTYPE = torch::kFloat32;

struct FCValueNet : torch::nn::Module {
  FCValueNet(int obs_size, int hidden_layer_count, int hidden_layer_size);
  torch::Tensor forward(torch::Tensor x);

  std::vector<torch::nn::Linear> layers_;
};

class FCValueEstimator {
public:
  FCValueEstimator(const Simulation &sim, Index robot_idx,
                   const torch::Device &device, int batch_size = 32,
                   int epoch_count = 4, int ensemble_size = 6);
  int getObservationSize() const;
  void getObservation(const Simulation &sim, Eigen::Ref<VectorX> obs) const;
  void estimateValue(const MatrixX &obs, Eigen::Ref<VectorX> value_est) const;
  void train(const MatrixX &obs, const Eigen::Ref<const VectorX> &value);

private:
  torch::Tensor torchTensorFromEigenMatrix(const MatrixX &mat) const;
  torch::Tensor
  torchTensorFromEigenVector(const Eigen::Ref<const VectorX> &vec) const;
  void torchTensorToEigenVector(const torch::Tensor &tensor,
                                Eigen::Ref<VectorX> vec) const;

  int robot_idx_;
  torch::Device device_;
  int batch_size_;
  int epoch_count_;
  int dof_count_;
  std::vector<std::shared_ptr<FCValueNet>> nets_;
  std::vector<std::shared_ptr<torch::optim::Adam>> optimizers_;
};

} // namespace robot_design
