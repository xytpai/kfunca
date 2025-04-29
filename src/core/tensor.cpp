#include <iostream>
#include <iomanip>
#include <vector>

#include "tensor.h"
#include "exception.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "memory_engine.h"
#include "binary_ops.h"
#include "reduce_ops.h"

using namespace utils::memory;

Tensor::Tensor(std::vector<int64_t> &shape, ScalarType dtype) {
    CHECK_FAIL(shape.size() <= MAX_TENSOR_DIMS);
    dtype_ = dtype;
    dim_ = shape.size();
    numel_ = 1;
    for (int i = dim_ - 1; i >= 0; i--) {
        stride_[i] = numel_;
        numel_ *= shape[i];
        shape_[i] = shape[i];
    }
}

Tensor::Tensor(int64_t *shape, int ndim, ScalarType dtype, bool inverse) {
    CHECK_FAIL(ndim <= MAX_TENSOR_DIMS);
    dtype_ = dtype;
    dim_ = ndim;
    numel_ = 1;
    int is;
    for (int i = dim_ - 1; i >= 0; i--) {
        stride_[i] = numel_;
        if (!inverse)
            is = i;
        else
            is = dim_ - 1 - i;
        numel_ *= shape[is];
        shape_[i] = shape[is];
    }
}

void Tensor::copy_from_cpu_ptr(void *ptr) {
    dmemcpy_h2d(data_ptr(), ptr, storage_bytes());
}

void Tensor::copy_to_cpu_ptr(void *ptr) const {
    dmemcpy_d2h(ptr, data_ptr(), storage_bytes());
}

any_t Tensor::item(const std::vector<int64_t> &indices) const {
    auto offset_ = offset(indices);
    any_t buffer;
    DISPATCH_BASIC_TYPES(dtype(), "Tensor::item", [&]() {
        dmemcpy_d2h(
            (void *)(buffer.val),
            (void *)((char *)data_ptr() + offset_ * sizeof(scalar_t)),
            sizeof(scalar_t));
    });
    return buffer;
}

int64_t Tensor::offset(const std::vector<int64_t> &indices) const {
    CHECK_FAIL(indices.size() == dim_);
    int64_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_index += indices[i] * stride_[i];
    }
    return flat_index;
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
    dmemset_zeros(output.data_ptr(), output.storage_bytes());
    return output;
}

void print_tensor_(std::ostream &os, const Tensor &t, std::vector<int64_t> indices = {}, int dim = 0) {
    if (dim == t.dim()) {
        auto result_ = t.item(indices);
        DISPATCH_BASIC_TYPES(t.dtype(), "print_tensor_", [&]() {
            os << std::fixed << std::showpos << std::setprecision(5) << *reinterpret_cast<scalar_t *>(&result_);
        });
        return;
    }
    if (dim > 0) os << "\n";
    for (int i = -1; i < dim; i++) os << "  ";
    os << "[";
    int64_t ii;
    for (ii = 0; ii < std::min<int64_t>(t.shape(dim), 12); ii++) {
        if (ii > 0) os << ", ";
        indices.push_back(ii);
        print_tensor_(os, t, indices, dim + 1);
        indices.pop_back();
    }
    if (t.shape(dim) != 12 && ii == 12) {
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

Tensor Tensor::operator+(const Tensor &other) const {
    return gpu::add(*this, other);
}

Tensor Tensor::operator-(const Tensor &other) const {
    return gpu::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor &other) const {
    return gpu::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor &other) const {
    return gpu::div(*this, other);
}

Tensor Tensor::sum(int64_t reduce_dim) const {
    return gpu::sum(*this, reduce_dim);
}

Tensor Tensor::mean(int64_t reduce_dim) const {
    return gpu::mean(*this, reduce_dim);
}
