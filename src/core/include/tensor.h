#pragma once

#include <optional>

#include "tensor_impl.h"

#define MAX_TENSOR_DIMS 12

class Tensor;
Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device = 0);
Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse = false);
Tensor empty_like(const Tensor &self);
Tensor empty_strided(std::vector<int64_t> shape, std::vector<int64_t> strides, ScalarType dtype, int device);
Tensor empty_like_reduced(const Tensor &self, int dim, ScalarType dtype);
Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device = 0);
std::ostream &operator<<(std::ostream &os, const Tensor &t);

class GradFunction : public intrusive_ptr_target {
public:
    virtual std::vector<Tensor> backward(Tensor grad_output) = 0;
    std::vector<Tensor> inputs;
};

class Tensor {
private:
    intrusive_ptr<TensorImpl> impl_;
    intrusive_ptr<GradFunction> grad_fn_;
    friend Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device);
    friend Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse);
    friend Tensor empty_like(const Tensor &self);
    friend Tensor empty_strided(std::vector<int64_t> shape, std::vector<int64_t> strides, ScalarType dtype, int device);
    friend Tensor empty_like_reduced(const Tensor &self, int dim, ScalarType dtype);
    friend Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device);

public:
    Tensor() = default;
    Tensor(const Tensor &other) = default;
    Tensor &operator=(const Tensor &other) = default;
    Tensor(Tensor &&other) = default;
    Tensor &operator=(Tensor &&other) = default;

    TensorImpl *impl() const {
        return impl_.get();
    }
    int dim() const {
        return impl_.get()->dim();
    }
    int64_t shape(int d) const {
        return impl_.get()->shape(d);
    }
    dim_t &shape() {
        return impl_.get()->shape();
    }
    std::vector<int64_t> sizes() const {
        return impl_.get()->sizes();
    }
    std::vector<int64_t> strides() const {
        return impl_.get()->strides();
    }
    int64_t stride(int d) const {
        return impl_.get()->stride(d);
    }
    dim_t &stride() {
        return impl_.get()->stride();
    }
    ScalarType dtype() const {
        return impl_.get()->dtype();
    }
    int64_t numel() const {
        return impl_.get()->numel();
    }
    void *data_ptr() const {
        return impl_.get()->data_ptr();
    }
    template <typename T>
    T *data_ptr() const {
        return impl_.get()->data_ptr<T>();
    }
    size_t storage_bytes() const {
        return impl_.get()->storage_bytes();
    }
    size_t storage_ref_count() const {
        return impl_.get()->storage_ref_count();
    }
    size_t impl_ref_count() const {
        return impl_.ref_count();
    }
    int64_t storage_offset() const {
        return impl_.get()->storage_offset();
    }
    intrusive_ptr<TensorStorage> storage() const {
        return impl_.get()->storage();
    }
    intrusive_ptr<GradFunction> grad_fn() const {
        return grad_fn_;
    }
    bool defined() const {
        return impl_.get() && impl_.get()->defined();
    }
    bool has_grad_fn() const {
        return grad_fn_.get() != nullptr;
    }
    int device() const {
        return impl_.get()->device();
    }
    int64_t element_size_in_bytes() const {
        return impl_.get()->element_size_in_bytes();
    }
    std::string to_string() const {
        std::ostringstream oss;
        oss << *this;
        return oss.str();
    }
    bool is_contiguous() const {
        return impl_.get()->is_contiguous();
    }
    bool requires_grad() const {
        return impl_.get()->requires_grad();
    }
    void set_requires_grad(bool flag) {
        return impl_.get()->set_requires_grad(flag);
    }
    Tensor *grad() {
        return impl_.get()->grad_.get();
    }

    void set_grad_fn(GradFunction *fn);
    void update_grad(Tensor grad);
    void backward(Tensor grad_output);
    void copy_from_cpu_ptr(void *ptr);
    void copy_to_cpu_ptr(void *ptr) const;
    any_t item(const std::vector<int64_t> &indices) const;
    Tensor &fill_(const any_t &value);
    int64_t offset(const std::vector<int64_t> &indices) const;
    Tensor contiguous() const;
    Tensor as_strided(std::vector<int64_t> sizes, std::vector<int64_t> strides, int64_t storage_offset = 0) const;
    Tensor permute(const std::vector<int64_t> dims) const;
    Tensor slice(int64_t dim, std::optional<int64_t> start, std::optional<int64_t> end, int64_t step = 1) const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    Tensor view(std::vector<int64_t> sizes) const;
    bool can_use_32bit_indexing() const;
    std::vector<Tensor> split(std::vector<int64_t> indices, int64_t dim) const;

    Tensor _half() const;
    Tensor _bfloat16() const;
    Tensor _float() const;

    Tensor operator+(const Tensor &other) const;
    Tensor &operator+=(const Tensor &other);
    Tensor operator-(const Tensor &other) const;
    Tensor &operator-=(const Tensor &other);
    Tensor operator*(const Tensor &other) const;
    Tensor &operator*=(const Tensor &other);
    Tensor operator/(const Tensor &other) const;
    Tensor &operator/=(const Tensor &other);
    Tensor &copy_(const Tensor &other);
    Tensor sum(int64_t reduce_dim) const;
    Tensor mean(int64_t reduce_dim) const;
    std::tuple<Tensor, Tensor> sort(int64_t dim, bool descending) const;
    std::tuple<Tensor, Tensor> topk(int64_t k, int64_t dim, bool largest) const;
    std::tuple<Tensor, Tensor> mean_var(int64_t reduce_dim, bool take_sqrt) const;
    std::tuple<Tensor, Tensor> norm_stat(int64_t dim) const;
    Tensor &index_put_(const std::vector<Tensor> &indices, const Tensor &values);
};
