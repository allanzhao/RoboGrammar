#include <pybind11/pybind11.h>

namespace py = pybind11;

void initGraph(py::module &m);
void initProp(py::module &m);
void initRobot(py::module &m);

PYBIND11_MODULE(pyrobotdesign, m) {
  initGraph(m);
  initProp(m);
  initRobot(m);
}
