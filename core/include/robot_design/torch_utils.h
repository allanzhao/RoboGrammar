#pragma once

#include <Eigen/Dense>
#include <robot_design/types.h>
#include <torch/torch.h>
#include <type_traits>

namespace robot_design {

// Torch dtype corresponding to Scalar
constexpr torch::Dtype SCALAR_DTYPE =
    std::is_same<Scalar, double>::value ? torch::kFloat64 : torch::kFloat32;

// Torch dtype used internally
constexpr torch::Dtype TORCH_DTYPE = torch::kFloat32;

inline torch::Tensor torchTensorFromEigenMatrix(const Ref<const MatrixX> &mat) {
  // Create a row-major Torch tensor from a column-major Eigen matrix
  return torch::from_blob(const_cast<Scalar *>(mat.data()),
                          {mat.cols(), mat.rows()}, torch::dtype(SCALAR_DTYPE))
      .toType(TORCH_DTYPE);
}

inline torch::Tensor torchTensorFromEigenVector(const Ref<const VectorX> &vec) {
  return torch::from_blob(const_cast<Scalar *>(vec.data()), {vec.size()},
                          torch::dtype(SCALAR_DTYPE))
      .toType(TORCH_DTYPE);
}

inline void torchTensorToEigenVector(const torch::Tensor &tensor,
                                     Ref<VectorX> vec) {
  vec = Eigen::Map<VectorX>(tensor.cpu().toType(SCALAR_DTYPE).data<Scalar>(),
                            vec.size());
}

} // namespace robot_design
