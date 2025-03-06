#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "device.h"

namespace py = pybind11;

PYBIND11_MODULE(kfunca, m) {
    m.def("launcher_test", &launcher_test, "Get device info");
}
