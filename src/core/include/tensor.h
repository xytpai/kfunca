#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include "exception.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "device_allocator.h"

#define MAX_TENSOR_DIMS 12

using namespace utils::memory;

class Tensor;
Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device = 0);
Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse = false);
Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device = 0);
std::ostream &operator<<(std::ostream &os, const Tensor &t);

template <typename T, int vec_size>
struct d_array {
    T val[vec_size] = {0};
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
    bool equals(d_array<T, vec_size> &other) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            if (val[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
    bool equals(T (&other)[vec_size]) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            if (val[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
};

typedef d_array<int64_t, MAX_TENSOR_DIMS> dim_t;
using any_t = d_array<char, 256>;
std::ostream &operator<<(std::ostream &os, const dim_t &d);

class TensorStorage : public intrusive_ptr_target {
protected:
    size_t size_;
    int device_;
    DataPtr ptr_;

public:
    TensorStorage() :
        size_(0), device_(-1), ptr_() {
    }
    TensorStorage(size_t size, int device) :
        size_(size), device_(device) {
        ptr_ = DeviceAllocator::GetInstance()->allocate(size, device);
    }
    size_t size() const {
        return size_;
    }
    int device() const {
        return device_;
    }
    void *data_ptr() const {
        return ptr_.get();
    }
    template <typename T>
    T *data_ptr() const {
        return reinterpret_cast<T *>(ptr_.get());
    }
    bool defined() const {
        return static_cast<bool>(ptr_);
    }
};

class Tensor {
    int dim_;
    dim_t shape_;
    dim_t stride_;
    ScalarType dtype_;
    int64_t numel_;
    intrusive_ptr<TensorStorage> storage_;

    void new_storage_(int device) {
        size_t bytes = shape_[0] * stride_[0] * element_size(dtype_);
        auto ptr = new TensorStorage(bytes, device);
        storage_.unsafe_set_ptr(ptr);
    }
    friend Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device);
    friend Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse);
    friend Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device);

    Tensor(std::vector<int64_t> &shape, ScalarType dtype);
    Tensor(int64_t *shape, int ndim, ScalarType dtype, bool inverse);

public:
    Tensor() :
        storage_() {
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

    int dim() const {
        return dim_;
    }
    int64_t shape(int d) const {
        return shape_[d];
    }
    dim_t &shape() {
        return shape_;
    }
    std::vector<int64_t> sizes() const {
        std::vector<int64_t> vec(shape_.val, shape_.val + dim_);
        return vec;
    }
    int64_t stride(int d) const {
        return stride_[d];
    }
    dim_t &stride() {
        return stride_;
    }
    ScalarType dtype() const {
        return dtype_;
    }
    int64_t numel() const {
        return numel_;
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
    intrusive_ptr<TensorStorage> storage() const {
        return storage_;
    }
    bool defined() const {
        return storage_.get() != nullptr;
    }
    int device() const {
        return storage_.get()->device();
    }
    int64_t element_size_in_bytes() const {
        return element_size(dtype_);
    }
    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }

    void copy_from_cpu_ptr(void *ptr);
    void copy_to_cpu_ptr(void *ptr) const;
    any_t item(const std::vector<int64_t> &indices) const;
    int64_t offset(const std::vector<int64_t> &indices) const;

    Tensor operator+(const Tensor &other) const;
    Tensor &operator+=(const Tensor &other);
    Tensor operator-(const Tensor &other) const;
    Tensor &operator-=(const Tensor &other);
    Tensor operator*(const Tensor &other) const;
    Tensor &operator*=(const Tensor &other);
    Tensor operator/(const Tensor &other) const;
    Tensor &operator/=(const Tensor &other);
    Tensor sum(int64_t reduce_dim) const;
    Tensor mean(int64_t reduce_dim) const;
    std::tuple<Tensor, Tensor> mean_var(int64_t reduce_dim, bool take_sqrt) const;
};
