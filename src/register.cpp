#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

#include "device_info.h"
#include "tensor.h"
#include "scalar_type.h"

namespace py = pybind11;

PYBIND11_MODULE(kfunca, m) {
    m.def("device_info", &device_info, "Get device info");
    py::enum_<ScalarType>(m, "ScalarType")
        .value("Byte", ScalarType::Byte)
        .value("Char", ScalarType::Char)
        .value("Short", ScalarType::Short)
        .value("Int", ScalarType::Int)
        .value("Long", ScalarType::Long)
        .value("Float", ScalarType::Float)
        .value("Double", ScalarType::Double)
        .value("Bool", ScalarType::Bool)
        .export_values();
    m.def("zeros", &zeros, "Allocate empty tensor");
    py::class_<Tensor>(m, "Tensor")
        .def("defined", &Tensor::defined)
        .def("numel", &Tensor::numel)
        .def("dim", &Tensor::dim)
        .def("device", &Tensor::device)
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype);
}
