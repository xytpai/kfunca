#include <iostream>
#include <vector>

#include "tensor.h"
#include "tensor_storage.h"
#include "exception.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "launcher.h"

using namespace utils::memory;

void Tensor::copy_from_cpu_ptr(void *ptr) {
    auto l = Launcher::GetInstance();
    l->memcpy(data_ptr(), ptr, storage_bytes(), Launcher::COPY::H2D);
}

void Tensor::copy_to_cpu_ptr(void *ptr) const {
    auto l = Launcher::GetInstance();
    l->memcpy(ptr, data_ptr(), storage_bytes(), Launcher::COPY::D2H);
}

any_t Tensor::item(const std::vector<int64_t> &indices) const {
    auto l = Launcher::GetInstance();
    auto offset_ = offset(indices);
    any_t buffer;
    DISPATCH_BASIC_TYPES(dtype(), "Tensor::item", [&]() {
        l->memcpy(
            (void *)(buffer.val),
            (void *)((char *)data_ptr() + offset_ * sizeof(scalar_t)),
            sizeof(scalar_t),
            Launcher::COPY::D2H);
    });
    return buffer;
}

Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device) {
    Tensor output(shape, dtype);
    output.new_storage_(device);
    return output;
}

Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse) {
    Tensor output(shape, ndim, dtype, inverse);
    output.new_storage_(device);
    return output;
}

Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device) {
    Tensor output(shape, dtype);
    output.new_storage_(device);
    Launcher::GetInstance()->memset(output.data_ptr(), 0, output.storage_bytes());
    return output;
}

void print_tensor_(std::ostream &os, const Tensor &t, std::vector<int64_t> indices = {}, int dim = 0) {
    if (dim == t.dim()) {
        auto result_ = t.item(indices);
        DISPATCH_BASIC_TYPES(t.dtype(), "print_tensor_", [&]() {
            os << *reinterpret_cast<scalar_t *>(&result_);
        });
        return;
    }
    if (dim > 0) os << "\n";
    for (int i = -1; i < dim; i++) os << "  ";
    os << "[";
    int64_t ii;
    for (ii = 0; ii < std::min<int64_t>(t.shape(dim), 20); ii++) {
        if (ii > 0) os << ", ";
        indices.push_back(ii);
        print_tensor_(os, t, indices, dim + 1);
        indices.pop_back();
    }
    if (t.shape(dim) != 20 && ii == 20) {
        os << ", ";
        if (dim < t.dim() - 1) {
            os << "\n";
            for (int i = -2; i < dim; i++) os << "  ";
        }
        os << "...";
    }
    if (dim < t.dim() - 1) {
        os << "\n";
        for (int i = -1; i < dim; i++) os << "  ";
    }
    os << "]";
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
    t.data_ptr();
    os << "tensor(shape=[";
    for (int i = 0; i < t.dim(); ++i)
        os << t.shape(i) << ",";
    os << "\b], stride=[";
    for (int i = 0; i < t.dim(); ++i)
        os << t.stride(i) << ",";
    os << "\b], dtype=" << t.dtype();
    os << ", numel=" << t.numel() << ", dim=" << t.dim();
    os << ", device=" << t.device() << ") {\n";
    print_tensor_(os, t);
    os << "\n}";
    return os;
}
