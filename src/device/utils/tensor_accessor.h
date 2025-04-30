#pragma once

#include "tensor.h"
#include "launcher.h"
#include "exception.h"

#include <cstdint>
#include <type_traits>
#include <algorithm>

template <typename T, size_t N, typename index_t = int64_t>
class TensorAccessorBase {
public:
    HOST_DEVICE TensorAccessorBase(
        T *data,
        const index_t *sizes,
        const index_t *strides) :
        data_(data),
        sizes_(sizes), strides_(strides) {
    }
    HOST_DEVICE index_t stride(index_t i) const {
        return strides_[i];
    }
    HOST_DEVICE index_t size(index_t i) const {
        return sizes_[i];
    }
    HOST_DEVICE T *data() {
        return data_;
    }
    HOST_DEVICE const T *data() const {
        return data_;
    }

protected:
    T *data_;
    const index_t *sizes_;
    const index_t *strides_;
};

template <typename T, size_t N, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T, N, index_t> {
public:
    HOST_DEVICE TensorAccessor(
        T *data,
        const index_t *sizes,
        const index_t *strides) :
        TensorAccessorBase<T, N, index_t>(data, sizes, strides) {
    }
    HOST_DEVICE TensorAccessor<T, N - 1, index_t> operator[](index_t i) {
        return TensorAccessor<T, N - 1, index_t>(
            this->data_ + this->strides_[0] * i, this->sizes_ + 1, this->strides_ + 1);
    }
    HOST_DEVICE const TensorAccessor<T, N - 1, index_t> operator[](index_t i) const {
        return TensorAccessor<T, N - 1, index_t>(
            this->data_ + this->strides_[0] * i, this->sizes_ + 1, this->strides_ + 1);
    }
};

template <typename T, typename index_t>
class TensorAccessor<T, 1, index_t> : public TensorAccessorBase<T, 1, index_t> {
public:
    HOST_DEVICE TensorAccessor(
        T *data_,
        const index_t *sizes_,
        const index_t *strides_) :
        TensorAccessorBase<T, 1, index_t>(data_, sizes_, strides_) {
    }
    HOST_DEVICE T &operator[](index_t i) {
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        return this->data_[this->strides_[0] * i];
    }
    HOST_DEVICE const T &operator[](index_t i) const {
        return this->data_[this->strides_[0] * i];
    }
};

template <typename T, size_t N, typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
public:
    HOST GenericPackedTensorAccessorBase(
        T *data,
        const index_t *sizes,
        const index_t *strides) :
        data_(data) {
        std::copy(sizes, sizes + N, std::begin(sizes_));
        std::copy(strides, strides + N, std::begin(strides_));
    }
    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
    HOST GenericPackedTensorAccessorBase(
        T *data,
        const source_index_t *sizes,
        const source_index_t *strides) :
        data_(data) {
        for (size_t i = 0; i < N; i++) {
            sizes_[i] = sizes[i];
            strides_[i] = strides[i];
        }
    }
    HOST_DEVICE index_t stride(index_t i) const {
        return strides_[i];
    }
    HOST_DEVICE index_t size(index_t i) const {
        return sizes_[i];
    }
    HOST_DEVICE T *data() {
        return data_;
    }
    HOST_DEVICE const T *data() const {
        return data_;
    }

protected:
    T *data_;
    index_t sizes_[N];
    index_t strides_[N];
    HOST void bounds_check_(index_t i) const {
        CHECK_FAIL(
            0 <= i && i < index_t{N},
            "Index ",
            i,
            " is not within bounds of a tensor of dimension ",
            N);
    }
};

template <typename T, size_t N, typename index_t = int64_t>
class GenericPackedTensorAccessor : public GenericPackedTensorAccessorBase<T, N, index_t> {
public:
    HOST GenericPackedTensorAccessor(
        T *data,
        const index_t *sizes,
        const index_t *strides) :
        GenericPackedTensorAccessorBase<T, N, index_t>(data, sizes, strides) {
    }
    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
    HOST GenericPackedTensorAccessor(
        T *data,
        const source_index_t *sizes,
        const source_index_t *strides) :
        GenericPackedTensorAccessorBase<T, N, index_t>(data, sizes, strides) {
    }
    DEVICE TensorAccessor<T, N - 1, index_t> operator[](index_t i) {
        index_t *new_sizes = this->sizes_ + 1;
        index_t *new_strides = this->strides_ + 1;
        return TensorAccessor<T, N - 1, index_t>(this->data_ + this->strides_[0] * i, new_sizes, new_strides);
    }
    DEVICE const TensorAccessor<T, N - 1, index_t> operator[](index_t i) const {
        const index_t *new_sizes = this->sizes_ + 1;
        const index_t *new_strides = this->strides_ + 1;
        return TensorAccessor<T, N - 1, index_t>(this->data_ + this->strides_[0] * i, new_sizes, new_strides);
    }
};
