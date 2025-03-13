#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

#include "device_info.h"
#include "tensor.h"
#include "binary_ops.h"

namespace py = pybind11;

Tensor from_numpy(py::array array, int device) {
    py::buffer_info buf = array.request();
#define HANDLE_DTYPE(cpp_type, scalar_type, ...)                         \
    if (buf.format == py::format_descriptor<cpp_type>::format()) {       \
        auto ptr = static_cast<cpp_type *>(buf.ptr);                     \
        auto output = empty(buf.shape, ScalarType::scalar_type, device); \
        output.copy_from_cpu_ptr((void *)ptr);                           \
        return output;                                                   \
    }
    FORALL_BASIC_SCALAR_TYPES(HANDLE_DTYPE)
    throw std::runtime_error("Unsupported dtype in from_numpy()");
#undef HANDLE_DTYPE
}

py::array to_numpy(const Tensor &t) {
#define HANDLE_DTYPE(cpp_type, scalar_type, ...)     \
    case ScalarType::scalar_type: {                  \
        py::array_t<cpp_type> array(t.sizes());      \
        py::buffer_info buf = array.request();       \
        auto ptr = static_cast<cpp_type *>(buf.ptr); \
        t.copy_to_cpu_ptr((void *)ptr);              \
        return array;                                \
    } break;
    switch (t.dtype()) {
        FORALL_BASIC_SCALAR_TYPES(HANDLE_DTYPE)
    default:
        throw std::runtime_error("Unsupported dtype in to_numpy()");
    }
#undef HANDLE_DTYPE
}

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
    m.def("from_numpy", from_numpy);
    m.def("to_numpy", to_numpy);
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
#define HANDLE_DTYPE(cpp_type, scalar_type, ...)               \
    case ScalarType::scalar_type: {                            \
        return py::cast(*reinterpret_cast<cpp_type *>(&data)); \
    } break;
            switch (self.dtype()) {
                FORALL_BASIC_SCALAR_TYPES(HANDLE_DTYPE)
            default:
                return py::none();
            }
#undef HANDLE_DTYPE
        })
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", &Tensor::operator*)
        .def("__truediv__", &Tensor::operator/);
}
