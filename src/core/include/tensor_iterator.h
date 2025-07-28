#pragma once

#include <iostream>
#include <ostream>
#include <vector>
#include <limits>
#include <memory>
#include <algorithm>

#include "tensor.h"
#include "exception.h"

struct SplitUntil32Bit;

struct TensorProp {
    bool is_output = false;
    bool is_read_write = false;
    bool will_resize = false;
};

class TensorIterator final {
    enum {
        MAX_TENSORS = 8,
    };

private:
    Tensor *tensors_[MAX_TENSORS];
    TensorProp tensor_props_[MAX_TENSORS];
    void *data_ptr_[MAX_TENSORS];

    int64_t shape_[MAX_TENSOR_DIMS];
    int64_t stride_bytes_[MAX_TENSORS][MAX_TENSOR_DIMS];
    int64_t perm_[MAX_TENSOR_DIMS];
    int64_t view_offsets_[MAX_TENSOR_DIMS];

    int common_device_ = -1;
    int num_outputs_ = 0;
    int num_inputs_ = 0;
    int num_tensors_ = 0;
    int ndim_ = 0;
    bool resize_outputs_ = true;
    bool accumulate_ = false;
    bool final_output_ = true;
    bool is_reduction_ = false;
    bool check_mem_overlap_ = true;
    int64_t reduce_dim_ = 0;
    ScalarType common_dtype_ = ScalarType::Undefined;

    void check_and_compute_common_device();
    void check_and_compute_dim();
    void compute_common_dtype();
    void allocate_reduction_output_if_need();
    void mark_outputs();
    void check_mem_overlaps();
    void compute_broadcasted_shape();
    void mark_resize_outputs();
    void compute_broadcasted_strides();
    void permute_dimensions();
    void reorder_dimensions();
    void allocate_outputs();
    void coalesce_dimensions();
    void update_data_pointers();

public:
    friend std::ostream &operator<<(std::ostream &os, const TensorIterator &iter);

    TensorIterator();
    TensorIterator &add_output(Tensor &output);
    TensorIterator &add_input(const Tensor &input);
    TensorIterator &add_output(Tensor &&output) = delete;
    TensorIterator &add_input(Tensor &&input) = delete;

    int64_t numel() const;
    int64_t num_output_elements() const;
    int num_reduce_dims() const;
    bool can_use_32bit_indexing() const;
    bool has_contiguous_first_dim() const;
    bool is_contiguous() const;
    int get_dim_to_split() const;
    bool is_dim_reduced(int dim) const;
    void narrow(int dim, int64_t start, int64_t size);
    std::unique_ptr<TensorIterator> split(int dim);
    SplitUntil32Bit with_32bit_indexing() const;

    TensorIterator &resize_outputs(bool flag) {
        resize_outputs_ = flag;
        return *this;
    }

    TensorIterator &check_mem_overlap(bool flag) {
        check_mem_overlap_ = flag;
        return *this;
    }

    TensorIterator &build();
    TensorIterator &build_for_loops();
    TensorIterator &build_for_reduce(int64_t reduce_dim);

    int ntensors() const {
        return num_tensors_;
    }

    int noutputs() const {
        return num_outputs_;
    }

    int ninputs() const {
        return num_inputs_;
    }

    int device(int arg = 0) const {
        return tensors_[arg]->device();
    }

    int64_t view_offsets(int arg) const {
        return view_offsets_[arg];
    }

    const Tensor &tensor(int arg) const {
        return *tensors_[arg];
    }

    int64_t shape(int dim) const {
        dim = maybe_wrap_dim(dim, ndim_);
        return shape_[dim];
    }

    int64_t *shape() {
        return shape_;
    }

    int64_t stride_bytes(int arg, int dim) const {
        return stride_bytes_[arg][dim];
    }

    int64_t *strides(int arg) {
        return stride_bytes_[arg];
    }

    int dim() const {
        return ndim_;
    }

    int ndim() const {
        return ndim_;
    }

    int64_t perm(int dim) const {
        return perm_[dim];
    }

    Tensor &outputs(int arg) {
        return *tensors_[arg];
    }

    /// If the kernel should accumulate into the output. Only relevant for reductions
    bool should_accumulate() const {
        return accumulate_;
    }

    bool is_final_output() const {
        return final_output_;
    }

    ScalarType input_dtype(int arg = 0) const {
        return tensors_[num_outputs_ + arg]->dtype();
    }

    ScalarType dtype(int arg = 0) const {
        return tensors_[arg]->dtype();
    }

    ScalarType common_dtype() const {
        CHECK_FAIL(
            common_dtype_ != ScalarType::Undefined,
            "Queried for invalid common dtype!");
        return common_dtype_;
    }

    void *data_ptr(int arg) const {
        return data_ptr_[arg];
    }

    int64_t element_size_in_bytes(int arg) const {
        return tensors_[arg]->element_size_in_bytes();
    }
};

struct SplitUntil32Bit {
    struct iterator {
        iterator() {
        }
        iterator(const TensorIterator &iter) {
            vec.emplace_back(new TensorIterator(iter));
            vec.emplace_back(nullptr); // ++ first pops the last element
            ++(*this);
        }
        iterator(iterator &&) = default;
        TensorIterator &operator*() const {
            return *vec.back();
        }
        iterator &operator++() {
            vec.pop_back();
            while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
                auto &iter = *vec.back();
                int64_t split_dim = iter.get_dim_to_split();
                vec.emplace_back(iter.split(split_dim));
            }
            return *this;
        }
        bool operator==(const iterator &other) const {
            // two iterators are equal if they are the same object or they're both empty
            return this == &other || (vec.empty() && other.vec.empty());
        }
        // needed for C++11 range-based for loop
        bool operator!=(const iterator &other) const {
            return !(*this == other);
        }
        /// stack of TensorIterators to be split
        std::vector<std::unique_ptr<TensorIterator>> vec;
    };
    SplitUntil32Bit(const TensorIterator &iter) :
        iter(iter) {
    }
    iterator begin() const {
        return iterator(iter);
    }
    iterator end() const {
        return iterator();
    }

private:
    const TensorIterator &iter;
};

std::ostream &operator<<(std::ostream &os, const TensorIterator &iter);
