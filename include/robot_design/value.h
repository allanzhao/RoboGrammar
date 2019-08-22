#pragma once

#include <memory>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <torch/torch.h>
#include <vector>

namespace robot_design {

using Eigen::Ref;

struct FCValueNet : torch::nn::Module {
  FCValueNet(int obs_size, int hidden_layer_count, int hidden_layer_size);
  torch::Tensor forward(torch::Tensor x);

  std::vector<torch::nn::Linear> layers_;
};

class FCValueEstimator {
public:
  FCValueEstimator();
  int getObservationSize(const Simulation &sim) const;
  void getObservation(const Simulation &sim, Ref<VectorX> obs) const;
  void estimateValue(const MatrixX &obs, Ref<VectorX> value) const;

private:
  std::shared_ptr<FCValueNet> net_;
};

}  // namespace robot_design
