#pragma once

#include <torch/torch.h>
#include <vector>

namespace robot_design {

struct FullyConnectedCritic : torch::nn::Module {
  FullyConnectedCritic(int obs_size, int hidden_layer_count,
                       int hidden_layer_size);
  torch::Tensor forward(torch::Tensor x);

  std::vector<torch::nn::Linear> layers_;
};

}  // namespace robot_design
