#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>

#include "device_info.h"
#include "tensor.h"
#include "binary_ops.h"
#include "gemm_ops.h"
#include "nn_ops.h"
#include "device_allocator.h"

namespace py = pybind11;

#define FORALL_NUMPY_BASIC_SCALAR_TYPES(_, ...) \
    _(bool, Bool, __VA_ARGS__)     /* 0 */      \
    _(uint8_t, Byte, __VA_ARGS__)  /* 1 */      \
    _(int8_t, Char, __VA_ARGS__)   /* 2 */      \
    _(int16_t, Short, __VA_ARGS__) /* 3 */      \
    _(int, Int, __VA_ARGS__)       /* 4 */      \
    _(int64_t, Long, __VA_ARGS__)  /* 5 */      \
    _(float, Float, __VA_ARGS__)   /* 6 */      \
    _(double, Double, __VA_ARGS__) /* 7 */

Tensor from_numpy(py::array array, int device) {
    py::buffer_info buf = array.request();
#define HANDLE_DTYPE(cpp_type, scalar_type, ...)                         \
    if (buf.format == py::format_descriptor<cpp_type>::format()) {       \
        auto ptr = static_cast<cpp_type *>(buf.ptr);                     \
        auto output = empty(buf.shape, ScalarType::scalar_type, device); \
        output.copy_from_cpu_ptr((void *)ptr);                           \
        return output;                                                   \
    }
    FORALL_NUMPY_BASIC_SCALAR_TYPES(HANDLE_DTYPE)
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
        FORALL_NUMPY_BASIC_SCALAR_TYPES(HANDLE_DTYPE)
    default:
        throw std::runtime_error("Unsupported dtype in to_numpy()");
    }
#undef HANDLE_DTYPE
}

PYBIND11_MODULE(kfunca, m) {
    m.def("device_info", &device_info);
    m.def("memstat", []() {
        DeviceAllocator::GetInstance()->print();
    });
    py::enum_<ScalarType>(m, "dtype")
        .value("byte", ScalarType::Byte)
        .value("char", ScalarType::Char)
        .value("short", ScalarType::Short)
        .value("int", ScalarType::Int)
        .value("long", ScalarType::Long)
        .value("half", ScalarType::Half)
        .value("float", ScalarType::Float)
        .value("double", ScalarType::Double)
        .value("bool", ScalarType::Bool)
        .export_values();
    m.def("empty", [](std::vector<int64_t> shape, ScalarType dtype, int device) {
        return empty(shape, dtype, device);
    });
    m.def("empty_like", [](const Tensor &self) {
        return empty_like(self);
    });
    m.def("from_numpy", from_numpy);
    m.def("to_numpy", to_numpy);
    m.def("zeros", &zeros);
    m.def("causal_attention", &gpu::causal_attention);
    m.def("gemm", &gpu::gemm);
    py::class_<Tensor>(m, "tensor")
        .def("__copy__", [](const Tensor &self) { return Tensor(self); })
        .def("__deepcopy__", [](const Tensor &self, py::dict) { return Tensor(self); })
        .def("__repr__", &Tensor::to_string)
        .def("defined", &Tensor::defined)
        .def("numpy", [](const Tensor &self) { return to_numpy(self); })
        .def("numel", &Tensor::numel)
        .def("dim", &Tensor::dim)
        .def("device", &Tensor::device)
        .def("shape", [](const Tensor &self, int64_t d) { return self.shape(d); })
        .def("sizes", [](const Tensor &self) { return self.sizes(); })
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
        .def("fill_", [](Tensor &self, double value) {
            return self.fill_(any_t{value});
        })
        .def("data_ptr", [](Tensor &self) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(self.data_ptr());
        })
        .def("storage_ref_count", &Tensor::storage_ref_count)
        .def("contiguous", &Tensor::contiguous)
        .def("permute", &Tensor::permute)
        .def("__add__", &Tensor::operator+)
        .def("__add__", [](const Tensor &self, double scalar) {
            return self + empty_like(self).fill_(any_t{scalar});
        })
        .def("__iadd__", &Tensor::operator+=)
        .def("__iadd__", [](Tensor &self, double scalar) {
            self += empty_like(self).fill_(any_t{scalar});
            return self;
        })
        .def("__sub__", &Tensor::operator-)
        .def("__sub__", [](const Tensor &self, double scalar) {
            return self - empty_like(self).fill_(any_t{scalar});
        })
        .def("__isub__", &Tensor::operator-=)
        .def("__isub__", [](Tensor &self, double scalar) {
            self -= empty_like(self).fill_(any_t{scalar});
            return self;
        })
        .def("__mul__", &Tensor::operator*)
        .def("__mul__", [](const Tensor &self, double scalar) {
            return self * empty_like(self).fill_(any_t{scalar});
        })
        .def("__imul__", &Tensor::operator*=)
        .def("__imul__", [](Tensor &self, double scalar) {
            self *= empty_like(self).fill_(any_t{scalar});
            return self;
        })
        .def("__truediv__", &Tensor::operator/)
        .def("__truediv__", [](const Tensor &self, double scalar) {
            return self / empty_like(self).fill_(any_t{scalar});
        })
        .def("__itruediv__", &Tensor::operator/=)
        .def("__itruediv__", [](Tensor &self, double scalar) {
            self /= empty_like(self).fill_(any_t{scalar});
            return self;
        })
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("mean_var", &Tensor::mean_var)
        .def("norm_stat", &Tensor::norm_stat)
        .def("half", &Tensor::_half)
        .def("float", &Tensor::_float);
}
