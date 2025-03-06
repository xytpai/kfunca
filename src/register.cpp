#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "device_info.h"

namespace py = pybind11;

PYBIND11_MODULE(kfunca, m) {
    m.def("device_info", &device_info, "Get device info");
}
