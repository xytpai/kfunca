#include <iostream>
#include <iomanip>
#include <vector>

#include "tensor.h"
#include "tensor_shape.h"
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

Tensor Tensor::as_strided(std::vector<int64_t> sizes,
                          std::vector<int64_t> strides, int64_t storage_offset) const {
    auto ndim = sizes.size();
    bool has_strides = strides.size() > 0;
    if (has_strides) {
        CHECK_FAIL(ndim == strides.size());
    } else {
        strides.reserve(ndim);
        int64_t cumprod = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = cumprod;
            cumprod *= sizes[i];
        }
    }
    // in-bounds check
    auto [min_offset, max_offset] = compute_offset_range(sizes, strides, ndim);
    min_offset += storage_offset;
    max_offset += storage_offset;
    CHECK_FAIL(min_offset >= 0);
    CHECK_FAIL(max_offset * element_size(dtype_) < this->storage_bytes());
    // create tensor view
    Tensor out(*this);
    out.dim_ = ndim;
    int64_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        out.shape_[i] = sizes[i];
        numel *= sizes[i];
        out.stride_[i] = strides[i];
    }
    out.numel_ = numel;
    out.is_contiguous_ = false;
    out.storage_offset_ = storage_offset;
    return out;
}

Tensor Tensor::permute(const std::vector<int64_t> dims) const {
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

Tensor Tensor::slice(int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end, int64_t step) const {
    auto &self = *this;
    int64_t ndim = self.dim();
    dim = maybe_wrap_dim(dim, ndim);
    auto sizes = self.sizes();
    auto strides = self.strides();
    int64_t start_val = start.has_value() ? start.value() : 0;
    int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
    CHECK_FAIL(step > 0, "slice step must be positive");
    if (start_val < 0) {
        start_val += sizes[dim];
    }
    if (end_val < 0) {
        end_val += sizes[dim];
    }
    if (start_val < 0) {
        start_val = 0;
    } else if (start_val >= sizes[dim]) {
        start_val = sizes[dim];
    }
    if (end_val < start_val) {
        end_val = start_val;
    } else if (end_val >= sizes[dim]) {
        end_val = sizes[dim];
    }
    auto storage_offset = self.storage_offset() + start_val * strides[dim];
    auto len = end_val - start_val;
    sizes[dim] = (len + step - 1) / step; // round-up
    strides[dim] *= step;
    Tensor result = self.as_strided(sizes, strides, storage_offset);
    return result;
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    auto &self = *this;
    int64_t ndim = self.dim();
    if (ndim == 0) {
        CHECK_FAIL(false, "select() cannot be applied to a 0-dim tensor.");
    }
    dim = maybe_wrap_dim(dim, ndim);
    auto size = self.shape(dim);
    if (size <= -1 - index || size <= index) {
        CHECK_FAIL(false);
    }
    if (index < 0) {
        index += size;
    }
    Tensor result;
    auto sizes = self.sizes();
    auto strides = self.strides();
    auto storage_offset = self.storage_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
    result = self.as_strided(sizes, strides, storage_offset);
    return result;
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    auto &self = *this;
    CHECK_FAIL(self.dim() > 0, "narrow() cannot be applied to a 0-dim tensor.");
    CHECK_FAIL(length >= 0, "narrow(): length must be non-negative.");
    auto cur_size = self.shape(dim);
    if (start < 0) {
        start = start + cur_size;
    }
    CHECK_FAIL(
        start <= cur_size - length,
        "start (",
        start,
        ") + length (",
        length,
        ") exceeds dimension size (",
        cur_size,
        ").");
    return this->slice(dim, start, start + length, 1);
}

Tensor Tensor::view(std::vector<int64_t> sizes) const {
    CHECK_FAIL(this->is_contiguous_);
    int64_t cumprod = 1;
    bool has_neg_dim = false;
    int64_t neg_dim = -1;
    for (auto i = 0; i < sizes.size(); ++i) {
        auto size = sizes[i];
        if (size < 0) {
            CHECK_FAIL(has_neg_dim == false);
            has_neg_dim = true;
            neg_dim = i;
        } else {
            cumprod *= size;
        }
    }
    if (has_neg_dim) {
        sizes[neg_dim] = this->numel_ / cumprod;
        cumprod *= sizes[neg_dim];
    }
    CHECK_FAIL(cumprod == this->numel_);
    return this->as_strided(sizes, {});
}

bool Tensor::can_use_32bit_indexing() const {
    int64_t max_value = std::numeric_limits<int32_t>::max();
    if (this->numel() > max_value) {
        return false;
    }
    int64_t max_offset = 1;
    for (int d = 0; d < dim_; ++d) {
        max_offset += (shape_[d] - 1) * stride_[d] * element_size(dtype_);
    }
    if (max_offset > max_value) {
        return false;
    }
    return true;
}

std::vector<Tensor> Tensor::split(std::vector<int64_t> indices, int64_t dim) const {
    return gpu::tensor_split(*this, indices, dim);
}

Tensor Tensor::_half() const {
    return gpu::convert(*this, ScalarType::Half);
}

Tensor Tensor::_bfloat16() const {
    return gpu::convert(*this, ScalarType::BFloat16);
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
    os << "\b], storage_offset=" << t.storage_offset();
    os << ", dtype=" << t.dtype();
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

std::tuple<Tensor, Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest) const {
    return gpu::topk(*this, k, dim, largest);
}

std::tuple<Tensor, Tensor> Tensor::mean_var(int64_t reduce_dim, bool take_sqrt) const {
    return gpu::mean_var(*this, reduce_dim, take_sqrt);
}

std::tuple<Tensor, Tensor> Tensor::norm_stat(int64_t dim) const {
    return gpu::norm_stat(*this, dim);
}
