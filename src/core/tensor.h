#pragma once

#include <iostream>
#include <vector>

#include "exception.h"
#include "array.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "launcher.h"

#define MAX_TENSOR_DIMS 12

using namespace memory;

class Tensor;
Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device = 0);
Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse = false);
Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device = 0);

typedef memory::array<int64_t, MAX_TENSOR_DIMS> dim_t;

class Tensor {
    int dim_;
    dim_t shape_;
    dim_t stride_;
    ScalarType dtype_;
    int64_t numel_;

    intrusive_ptr<memory::TensorStorage> storage_;

    void new_storage_(int device) {
        size_t bytes = shape_[0] * stride_[0] * element_size(dtype_);
        auto ptr = new memory::TensorStorage(bytes, device);
        storage_.unsafe_set_ptr(ptr);
    }
    friend Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device);
    friend Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse);
    friend Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device);

public:
    Tensor(std::vector<int64_t> &shape, ScalarType dtype) :
        dtype_(dtype) {
        CHECK_FAIL(shape.size() <= MAX_TENSOR_DIMS);
        dim_ = shape.size();
        numel_ = 1;
        for (int i = dim_ - 1; i >= 0; i--) {
            stride_[i] = numel_;
            numel_ *= shape[i];
            shape_[i] = shape[i];
        }
    }
    Tensor(int64_t *shape, int ndim, ScalarType dtype, bool inverse) :
        dtype_(dtype) {
        CHECK_FAIL(ndim <= MAX_TENSOR_DIMS);
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
    Tensor(const Tensor &other) :
        dim_(other.dim_), shape_(other.shape_), stride_(other.stride_),
        dtype_(other.dtype_), numel_(other.numel_),
        storage_(other.storage_) {
    }
    Tensor &operator=(const Tensor &other) {
        dim_ = other.dim_;
        shape_ = other.shape_;
        stride_ = other.stride_;
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        storage_ = other.storage_;
        return *this;
    }
    Tensor(Tensor &&other) = default;
    Tensor &operator=(Tensor &&other) = default;
    Tensor() :
        storage_() {
    }
    bool defined() const {
        return storage_.get() != nullptr;
    }
    int64_t numel() const {
        return numel_;
    }
    int dim() const {
        return dim_;
    }
    int device() const {
        return storage_.get()->device();
    }
    int64_t shape(int d) const {
        return shape_[d];
    }
    int64_t stride(int d) const {
        return stride_[d];
    }
    void *data_ptr() const {
        return storage_.get()->data_ptr();
    }
    size_t storage_bytes() const {
        return storage_.get()->size();
    }
    size_t storage_ref_count() const {
        return storage_.ref_count();
    }
    intrusive_ptr<memory::TensorStorage> storage() const {
        return storage_;
    }
    ScalarType dtype() const {
        return dtype_;
    }
    dim_t &stride() {
        return stride_;
    }
    int64_t element_size_in_bytes() const {
        return element_size(dtype_);
    }
    void copy_from_cpu_ptr(void *ptr) {
        auto l = Launcher::GetInstance();
        l->memcpy(data_ptr(), ptr, storage_bytes(), Launcher::Direction::H2D);
    }
    void copy_to_cpu_ptr(void *ptr) {
        auto l = Launcher::GetInstance();
        l->memcpy(ptr, data_ptr(), storage_bytes(), Launcher::Direction::D2H);
    }
};

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

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
    os << "Tensor(shape=[";
    for (int i = 0; i < t.dim(); ++i)
        os << t.shape(i) << ",";
    os << "\b], stride=[";
    for (int i = 0; i < t.dim(); ++i)
        os << t.stride(i) << ",";
    os << "\b], dtype=" << t.dtype();
    os << ", numel=" << t.numel() << ", dim=" << t.dim();
    os << ", device=" << t.device() << ")";
    return os;
}
