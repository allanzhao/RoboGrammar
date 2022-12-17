#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <robot_design/optim.h>

namespace py = pybind11;
namespace rd = robot_design;

void initOptim(py::module &m) {
  py::class_<rd::InputSampler, std::shared_ptr<rd::InputSampler>>(
      m, "InputSampler")
      .def("sample_input_sequence", &rd::InputSampler::sampleInputSequence);

  py::class_<rd::DefaultInputSampler, rd::InputSampler,
             std::shared_ptr<rd::DefaultInputSampler>>(m, "DefaultInputSampler")
      .def(py::init<>())
      .def(py::init<rd::Scalar, rd::Scalar>())
      .def_readwrite("history_std_dev",
                     &rd::DefaultInputSampler::history_std_dev_)
      .def_readwrite("warm_start_std_dev",
                     &rd::DefaultInputSampler::warm_start_std_dev_);

  py::class_<rd::ConstantInputSampler, rd::InputSampler,
             std::shared_ptr<rd::ConstantInputSampler>>(m,
                                                        "ConstantInputSampler")
      .def(py::init<>())
      .def_readwrite("samples", &rd::ConstantInputSampler::samples_);

  py::class_<rd::MPPIOptimizer, std::shared_ptr<rd::MPPIOptimizer>>(
      m, "MPPIOptimizer")
      // Only SumOfSquaresObjective and DotProductObjective are supported by the
      // Python bindings for now
      .def(py::init<rd::Scalar, rd::Scalar, int, int, int, int, int,
                    unsigned int, const rd::MakeSimFunction &,
                    const rd::SumOfSquaresObjective &,
                    const std::shared_ptr<const rd::ValueEstimator> &,
                    const std::shared_ptr<const rd::InputSampler> &>())
      .def(py::init<rd::Scalar, rd::Scalar, int, int, int, int, int,
                    unsigned int, const rd::MakeSimFunction &,
                    const rd::DotProductObjective &,
                    const std::shared_ptr<const rd::ValueEstimator> &,
                    const std::shared_ptr<const rd::InputSampler> &>())
      .def("update", &rd::MPPIOptimizer::update,
           py::call_guard<py::gil_scoped_release>())
      .def("advance", &rd::MPPIOptimizer::advance,
           py::call_guard<py::gil_scoped_release>())
      .def("get_sample_count", &rd::MPPIOptimizer::getSampleCount)
      .def("set_sample_count", &rd::MPPIOptimizer::setSampleCount)
      .def_readwrite("input_sequence", &rd::MPPIOptimizer::input_sequence_);

  py::class_<rd::SumOfSquaresObjective>(m, "SumOfSquaresObjective")
      .def(py::init<>())
      .def("__call__", &rd::SumOfSquaresObjective::operator())
      .def_readwrite("base_vel_ref", &rd::SumOfSquaresObjective::base_vel_ref_)
      .def_readwrite("base_vel_weight",
                     &rd::SumOfSquaresObjective::base_vel_weight_)
      .def_readwrite("power_weight", &rd::SumOfSquaresObjective::power_weight_);

  py::class_<rd::DotProductObjective>(m, "DotProductObjective")
      .def(py::init<>())
      .def("__call__", &rd::DotProductObjective::operator())
      .def_readwrite("base_dir_weight",
                     &rd::DotProductObjective::base_dir_weight_)
      .def_readwrite("base_up_weight",
                     &rd::DotProductObjective::base_up_weight_)
      .def_readwrite("base_vel_weight",
                     &rd::DotProductObjective::base_vel_weight_)
      .def_readwrite("power_weight", &rd::DotProductObjective::power_weight_);
}
