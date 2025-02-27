#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

template <typename T>
void process_numpy_array(py::array_t<T> arr) {
    std::cout << sizeof(T) << "\n";
    // 获取数据指针
    py::buffer_info buf_info = arr.request();
    double *ptr = (double *)buf_info.ptr;

    // 获取数组的形状
    size_t rows = buf_info.shape[0];
    size_t cols = buf_info.shape[1];

    // 打印数组的内容
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << ptr[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

PYBIND11_MODULE(kfunca, m) {
    m.def("process_numpy_array", &process_numpy_array<double>, "Process a numpy array");
    m.def("process_numpy_array", &process_numpy_array<float>, "Process a numpy array");
}
