#include "tensor_impl.h"

std::ostream &operator<<(std::ostream &os, const dim_t &d) {
    os << "dim_t:";
    for (int i = 0; i < MAX_TENSOR_DIMS; i++)
        os << d[i] << ", ";
    os << "\n";
    return os;
}

TensorImpl::TensorImpl(std::vector<int64_t> &shape, ScalarType dtype) :
    grad_(nullptr, &delete_nothing) {
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

TensorImpl::TensorImpl(std::vector<int64_t> &shape, std::vector<int64_t> &strides, ScalarType dtype) :
    grad_(nullptr, &delete_nothing) {
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

TensorImpl::TensorImpl(int64_t *shape, int ndim, ScalarType dtype, bool inverse) :
    grad_(nullptr, &delete_nothing) {
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

void TensorImpl::new_storage_(int device) {
    bool has_defined_storage = storage_.get() && storage_.get()->defined();
    CHECK_FAIL(!has_defined_storage);
    auto [min_offset, max_offset] = compute_offset_range<dim_t>(shape_, stride_, dim_);
    size_t offset_range = max_offset - min_offset + 1;
    size_t bytes = offset_range * element_size(dtype_);
    auto ptr = new TensorStorage(bytes, device);
    storage_.unsafe_set_ptr(ptr);
}

void TensorImpl::as_strided_(std::vector<int64_t> sizes, std::vector<int64_t> strides, int64_t storage_offset) {
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
    dim_ = ndim;
    int64_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        shape_[i] = sizes[i];
        numel *= sizes[i];
        stride_[i] = strides[i];
    }
    numel_ = numel;
    is_contiguous_ = false;
    storage_offset_ = storage_offset;
    // TODO: remove it
    for (int i = ndim; i < MAX_TENSOR_DIMS; i++) {
        shape_[i] = 0;
        stride_[i] = 0;
    }
}
