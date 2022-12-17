#pragma once

#include <Eigen/Dense>
#include <memory>
#include <robot_design/sim.h>
#include <robot_design/types.h>
#include <vector>

namespace robot_design {

using Eigen::Ref;

class ValueEstimator {
public:
  virtual ~ValueEstimator() {}
  virtual int getObservationSize() const = 0;
  virtual void getObservation(
      const Simulation &sim, Ref<VectorX> obs) const = 0;
  virtual void estimateValue(
      const MatrixX &obs, Ref<VectorX> value_est) const = 0;
  virtual void train(const MatrixX &obs, const Ref<const VectorX> &value) = 0;
};

class NullValueEstimator : public ValueEstimator {
public:
  NullValueEstimator() {}
  virtual ~NullValueEstimator() {}
  virtual int getObservationSize() const override { return 0; }
  virtual void getObservation(
      const Simulation &sim, Ref<VectorX> obs) const override {}
  virtual void estimateValue(
      const MatrixX &obs, Ref<VectorX> value_est) const override {
    value_est = VectorX::Zero(obs.cols());
  }
  virtual void train(
      const MatrixX &obs, const Ref<const VectorX> &value) override {}
};

} // namespace robot_design
