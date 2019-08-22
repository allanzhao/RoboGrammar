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
  FCValueEstimator(const Simulation &sim, const torch::Device &device);
  int getObservationSize(const Simulation &sim) const;
  void getObservation(const Simulation &sim, Eigen::Ref<VectorX> obs) const;
  void estimateValue(const MatrixX &obs, Eigen::Ref<VectorX> value_est) const;

private:
  torch::Tensor torchTensorFromEigen(const MatrixX &mat) const;
  void torchTensorToEigen(const torch::Tensor &tensor,
                          Eigen::Ref<VectorX> vec) const;

  torch::Device device_;
  std::shared_ptr<FCValueNet> net_;
};

}  // namespace robot_design
