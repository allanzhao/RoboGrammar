#include <robot_design/nnet.h>
#include <sstream>

namespace robot_design {

FullyConnectedCritic::FullyConnectedCritic(int obs_size,
                                           int hidden_layer_count,
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

torch::Tensor FullyConnectedCritic::forward(torch::Tensor x) {
  for (int i = 0; i < layers_.size() - 1; ++i) {
    x = torch::relu(layers_[i]->forward(x));
  }
  return layers_.back()->forward(x);  // No activation after last layer
}

}  // namespace robot_design
