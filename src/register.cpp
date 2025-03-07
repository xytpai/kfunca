#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

#include "device_info.h"
#include "tensor.h"
#include "scalar_type.h"

namespace py = pybind11;

PYBIND11_MODULE(kfunca, m) {
    m.def("device_info", &device_info);
    py::enum_<ScalarType>(m, "dtype")
        .value("byte", ScalarType::Byte)
        .value("char", ScalarType::Char)
        .value("short", ScalarType::Short)
        .value("int", ScalarType::Int)
        .value("long", ScalarType::Long)
        .value("float", ScalarType::Float)
        .value("double", ScalarType::Double)
        .value("bool", ScalarType::Bool)
        .export_values();
    m.def("empty", [](std::vector<int64_t> shape, ScalarType dtype, int device) {
        return empty(shape, dtype, device);
    });
    m.def("zeros", &zeros);
    py::class_<Tensor>(m, "tensor")
        .def("__repr__", &Tensor::to_string)
        .def("defined", &Tensor::defined)
        .def("numel", &Tensor::numel)
        .def("dim", &Tensor::dim)
        .def("device", &Tensor::device)
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype)
        .def("item", [](Tensor &self, std::vector<int64_t> indices) -> py::object {
            auto data = self.item(indices);
            switch (self.dtype()) {
            case ScalarType::Byte:
                return py::cast(*reinterpret_cast<unsigned char *>(&data));
            case ScalarType::Char:
                return py::cast(*reinterpret_cast<char *>(&data));
            case ScalarType::Short:
                return py::cast(*reinterpret_cast<short *>(&data));
            case ScalarType::Int:
                return py::cast(*reinterpret_cast<int *>(&data));
            case ScalarType::Long:
                return py::cast(*reinterpret_cast<long *>(&data));
            case ScalarType::Float:
                return py::cast(*reinterpret_cast<float *>(&data));
            case ScalarType::Double:
                return py::cast(*reinterpret_cast<double *>(&data));
            case ScalarType::Bool:
                return py::cast(*reinterpret_cast<bool *>(&data));
            default:
                return py::none();
            }
        });
}
