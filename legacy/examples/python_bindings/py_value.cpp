#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <robot_design/value.h>
#include <stdexcept>

namespace py = pybind11;
namespace rd = robot_design;

void initValue(py::module &m) {
  py::class_<rd::ValueEstimator, std::shared_ptr<rd::ValueEstimator>>(
      m, "ValueEstimator")
      .def("get_observation_size", &rd::ValueEstimator::getObservationSize)
      .def("get_observation", &rd::ValueEstimator::getObservation)
      .def("estimate_value", &rd::ValueEstimator::estimateValue)
      .def("train", &rd::ValueEstimator::train);

  py::class_<rd::NullValueEstimator, rd::ValueEstimator,
             std::shared_ptr<rd::NullValueEstimator>>(m, "NullValueEstimator")
      .def(py::init<>());
}
