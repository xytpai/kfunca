#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "device_info.h"

namespace py = pybind11;

PYBIND11_MODULE(kfunca, m) {
    m.def("device_property", &device_property, "Get device info");
    m.def("global_memory_bandwidth", &global_memory_bandwidth, "Get global_memory_bandwidth");
}
