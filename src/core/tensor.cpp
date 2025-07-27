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
#include "unary_ops.h"
#include "nullary_ops.h"
#include "reduce_ops.h"
#include "sort_ops.h"
#include "norm_ops.h"
#include "accumulate_type.h"

using namespace utils::memory;

void Tensor::new_storage_(int device) {
    auto [min_offset, max_offset] = compute_offset_range<dim_t>(shape_, stride_, dim_);
    size_t offset_range = max_offset - min_offset + 1;
    size_t bytes = offset_range * element_size(dtype_);
    auto ptr = new TensorStorage(bytes, device);
    storage_.unsafe_set_ptr(ptr);
}

std::ostream &operator<<(std::ostream &os, const dim_t &d) {
    os << "dim_t:";
    for (int i = 0; i < MAX_TENSOR_DIMS; i++)
        os << d[i] << ", ";
    os << "\n";
    return os;
}

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

Tensor::Tensor(std::vector<int64_t> &shape, std::vector<int64_t> &strides, ScalarType dtype) {
    CHECK_FAIL(shape.size() <= MAX_TENSOR_DIMS);
    CHECK_FAIL(strides.size() <= MAX_TENSOR_DIMS);
    dtype_ = dtype;
    dim_ = shape.size();
    CHECK_FAIL(dim_ == strides.size());
    numel_ = 1;
    for (int i = dim_ - 1; i >= 0; i--) {
        stride_[i] = strides[i];
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

Tensor &Tensor::fill_(const any_t &value) {
    return gpu::fill_(*this, value);
}

int64_t Tensor::offset(const std::vector<int64_t> &indices) const {
    CHECK_FAIL(indices.size() == dim_);
    int64_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_index += indices[i] * stride_[i];
    }
    return flat_index;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous_)
        return *this;
    return gpu::clone(*this);
}

Tensor Tensor::as_strided(const std::vector<int64_t> sizes, const std::vector<int64_t> strides) {
    Tensor out(*this);
    bool is_strides_empty = strides.size() == 0;
    for (int i = 0; i < dim_; i++) {
        out.shape_[i] = sizes[i];
        if (!is_strides_empty)
            out.stride_[i] = strides[i];
    }
    out.is_contiguous_ = false;
    return out;
}

Tensor Tensor::permute(const std::vector<int64_t> dims) {
    const auto ndim = dim_;
    CHECK_FAIL(ndim == dims.size());
    auto new_sizes = std::vector<int64_t>(ndim);
    auto new_strides = std::vector<int64_t>(ndim);
    std::vector<bool> seen_dims(ndim);
    for (int i = 0; i < ndim; i++) {
        int d = maybe_wrap_dim(dims[i], ndim);
        CHECK_FAIL(!seen_dims[d], "permute(): duplicate dims are not allowed.");
        seen_dims[d] = true;
        new_sizes[i] = this->shape_[d];
        new_strides[i] = this->stride_[d];
    }
    return as_strided(new_sizes, new_strides);
}

Tensor Tensor::_half() const {
    return gpu::convert(*this, ScalarType::Half);
}

Tensor Tensor::_float() const {
    return gpu::convert(*this, ScalarType::Float);
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

Tensor empty_like(const Tensor &self) {
    auto sizes = self.sizes();
    Tensor output(sizes, self.dtype());
    output.new_storage_(self.device());
    return output;
}

Tensor empty_strided(std::vector<int64_t> shape, std::vector<int64_t> strides, ScalarType dtype, int device) {
    Tensor output(shape, strides, dtype);
    output.new_storage_(device);
    return output;
}

Tensor empty_like_reduced(const Tensor &self, int dim, ScalarType dtype) {
    auto sizes = self.sizes();
    if (dim >= 0) {
        sizes[dim] = 1;
    }
    Tensor output(sizes, dtype);
    output.new_storage_(self.device());
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
            using acc_t = acc_type<scalar_t>;
            os << std::fixed << std::showpos << std::setprecision(5) << (acc_t)(*reinterpret_cast<scalar_t *>(&result_));
            os << std::noshowpos;
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

Tensor &Tensor::operator+=(const Tensor &other) {
    return gpu::add_(*this, other);
}

Tensor Tensor::operator-(const Tensor &other) const {
    return gpu::sub(*this, other);
}

Tensor &Tensor::operator-=(const Tensor &other) {
    return gpu::sub_(*this, other);
}

Tensor Tensor::operator*(const Tensor &other) const {
    return gpu::mul(*this, other);
}

Tensor &Tensor::operator*=(const Tensor &other) {
    return gpu::mul_(*this, other);
}

Tensor Tensor::operator/(const Tensor &other) const {
    return gpu::div(*this, other);
}

Tensor &Tensor::operator/=(const Tensor &other) {
    return gpu::div_(*this, other);
}

Tensor &Tensor::copy_(const Tensor &other) {
    return gpu::copy_(*this, other);
}

Tensor Tensor::sum(int64_t reduce_dim) const {
    return gpu::sum(*this, reduce_dim);
}

Tensor Tensor::mean(int64_t reduce_dim) const {
    return gpu::mean(*this, reduce_dim);
}

std::tuple<Tensor, Tensor> Tensor::sort(int64_t dim, bool descending) const {
    return gpu::sort(*this, dim, descending);
}

std::tuple<Tensor, Tensor> Tensor::mean_var(int64_t reduce_dim, bool take_sqrt) const {
    return gpu::mean_var(*this, reduce_dim, take_sqrt);
}

std::tuple<Tensor, Tensor> Tensor::norm_stat(int64_t dim) const {
    return gpu::norm_stat(*this, dim);
}
