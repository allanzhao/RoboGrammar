#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <robot_design/value.h>
#include <stdexcept>

namespace py = pybind11;
namespace rd = robot_design;

void initValue(py::module &m) {
  py::class_<rd::FCValueEstimator, std::shared_ptr<rd::FCValueEstimator>>(
      m, "FCValueEstimator")
      .def(py::init([](const rd::Simulation &sim, rd::Index robot_idx,
                       const std::string &device_name, int batch_size,
                       int epoch_count, int ensemble_size) {
        torch::DeviceType device_type;
        if (device_name == "cpu") {
          device_type = torch::kCPU;
        } else if (device_name == "cuda") {
          device_type = torch::kCUDA;
        } else {
          throw std::invalid_argument("Invalid device name");
        }
        return std::make_shared<rd::FCValueEstimator>(
            sim, robot_idx, torch::Device(device_type), batch_size, epoch_count,
            ensemble_size);
      }))
      .def("get_observation_size", &rd::FCValueEstimator::getObservationSize)
      .def("get_observation", &rd::FCValueEstimator::getObservation)
      .def("estimate_value", &rd::FCValueEstimator::estimateValue)
      .def("train", &rd::FCValueEstimator::train);
}
