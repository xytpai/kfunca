#pragma once

#include <iostream>
#include <vector>

#include "data_ptr.h"
#include "exception.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "device_allocator.h"
#include "memory_overlap.h"

#define MAX_TENSOR_DIMS 12

using namespace utils::memory;

inline int maybe_wrap_dim(int d, int ndim) {
    return d < 0 ? (ndim + d) % ndim : d;
}

template <typename T, int vec_size>
struct d_array {
    T val[vec_size] = {0};
    d_array() {
    }
    d_array(double value) {
        *reinterpret_cast<double *>(&val[0]) = value;
    }
    operator double() const {
        return *reinterpret_cast<double *>(const_cast<T *>(&val[0]));
    }
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

class Tensor;
using TensorDeleter = void (*)(Tensor *);
inline void delete_nothing(Tensor *) {
}

class TensorImpl : public intrusive_ptr_target {
private:
    int dim_;
    dim_t shape_;
    dim_t stride_;
    ScalarType dtype_;
    int64_t numel_;
    intrusive_ptr<TensorStorage> storage_;
    int64_t storage_offset_ = 0;
    bool is_contiguous_ = true;
    bool requires_grad_ = false;

public:
    TensorImpl(std::vector<int64_t> &shape, ScalarType dtype);
    TensorImpl(std::vector<int64_t> &shape, std::vector<int64_t> &strides, ScalarType dtype);
    TensorImpl(int64_t *shape, int ndim, ScalarType dtype, bool inverse);
    TensorImpl() :
        storage_(), grad_(nullptr, &delete_nothing) {
    }
    TensorImpl(const TensorImpl &other) :
        dim_(other.dim_), shape_(other.shape_), stride_(other.stride_),
        dtype_(other.dtype_), numel_(other.numel_),
        storage_(other.storage_), grad_(nullptr, &delete_nothing),
        storage_offset_(other.storage_offset_),
        is_contiguous_(other.is_contiguous_),
        requires_grad_(other.requires_grad_) {
    }
    TensorImpl &operator=(const TensorImpl &other) {
        dim_ = other.dim_;
        shape_ = other.shape_;
        stride_ = other.stride_;
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        storage_ = other.storage_;
        grad_ = nullptr;
        storage_offset_ = other.storage_offset_;
        is_contiguous_ = other.is_contiguous_;
        requires_grad_ = other.requires_grad_;
        return *this;
    }
    TensorImpl(TensorImpl &&other) = default;
    TensorImpl &operator=(TensorImpl &&other) = default;

    int dim() const {
        return dim_;
    }
    int64_t shape(int d) const {
        d = maybe_wrap_dim(d, dim_);
        return shape_[d];
    }
    dim_t &shape() {
        return shape_;
    }
    std::vector<int64_t> sizes() const {
        std::vector<int64_t> vec(shape_.val, shape_.val + dim_);
        return vec;
    }
    std::vector<int64_t> strides() const {
        std::vector<int64_t> vec(stride_.val, stride_.val + dim_);
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
        return (char *)storage_.get()->data_ptr() + storage_offset_ * element_size(dtype_);
    }
    template <typename T>
    T *data_ptr() const {
        return reinterpret_cast<T *>(storage_.get()->data_ptr()) + storage_offset_;
    }
    size_t storage_bytes() const {
        return storage_.get()->size();
    }
    size_t storage_ref_count() const {
        return storage_.ref_count();
    }
    int64_t storage_offset() const {
        return storage_offset_;
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
    bool is_contiguous() const {
        return is_contiguous_;
    }
    bool requires_grad() const {
        return requires_grad_;
    }
    void set_requires_grad(bool flag) {
        requires_grad_ = flag;
    }
    void new_storage_(int device);
    void as_strided_(std::vector<int64_t> sizes, std::vector<int64_t> strides, int64_t storage_offset = 0);

public:
    std::unique_ptr<Tensor, TensorDeleter> grad_;
};
