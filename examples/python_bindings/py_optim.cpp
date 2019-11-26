#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <robot_design/optim.h>

namespace py = pybind11;
namespace rd = robot_design;

void initOptim(py::module &m) {
  py::class_<rd::MPPIOptimizer, std::shared_ptr<rd::MPPIOptimizer>>(
      m, "MPPIOptimizer")
      // Only SumOfSquaresObjective is supported for now
      .def(py::init<rd::Scalar, rd::Scalar, int, int, int, int, int,
                    unsigned int, const rd::MakeSimFunction &,
                    const rd::SumOfSquaresObjective &,
                    const std::shared_ptr<const rd::FCValueEstimator> &>())
      .def("update", &rd::MPPIOptimizer::update,
           py::call_guard<py::gil_scoped_release>())
      .def("advance", &rd::MPPIOptimizer::advance,
           py::call_guard<py::gil_scoped_release>())
      .def_readwrite("input_sequence", &rd::MPPIOptimizer::input_sequence_);

  py::class_<rd::SumOfSquaresObjective>(m, "SumOfSquaresObjective")
      .def(py::init<>())
      .def("__call__", &rd::SumOfSquaresObjective::operator())
      .def_readwrite("base_vel_ref", &rd::SumOfSquaresObjective::base_vel_ref_)
      .def_readwrite("base_vel_weight",
                     &rd::SumOfSquaresObjective::base_vel_weight_);
}
