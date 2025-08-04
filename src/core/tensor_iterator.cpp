#include "tensor_iterator.h"
#include "memory_overlap.h"

void TensorIterator::check_and_compute_common_device() {
    common_device_ = -1;
    for (int i = 0; i < num_tensors_; i++) {
        auto tensor_i = tensors_[i];
        if (!tensor_i->defined()) continue;
        if (common_device_ == -1 && tensor_i->device() >= 0) {
            common_device_ = tensor_i->device();
        } else {
            CHECK_FAIL(tensor_i->device() == common_device_, "All defined tensors should in the same device");
        }
    }
}

void TensorIterator::check_and_compute_dim() {
    bool is_first = true;
    int first_dim;
    for (int i = 0; i < num_tensors_; ++i) {
        if (!tensors_[i]->defined()) continue;
        if (is_first) {
            first_dim = tensors_[i]->dim();
            is_first = false;
        } else {
            CHECK_FAIL(first_dim == tensors_[i]->dim(), "All defined tensors should in the same dim");
        }
    }
    ndim_ = first_dim;
}

ScalarType update_common_dtype(ScalarType a, ScalarType b) {
    if (is_floating_type(a) && is_floating_type(b)) {
        return a >= b ? a : b;
    } else if (is_floating_type(a) || is_floating_type(b)) {
        return is_floating_type(a) ? a : b;
    } else if (is_unsigned_int_type(a) && is_unsigned_int_type(b)) {
        return a >= b ? a : b;
    } else if (is_unsigned_int_type(a) || is_unsigned_int_type(b)) {
        return is_unsigned_int_type(a) ? b : a;
    } else {
        return a >= b ? a : b;
    }
}

void TensorIterator::compute_common_dtype() {
    common_dtype_ = ScalarType::Undefined;
    for (int i = num_outputs_; i < num_tensors_; i++) {
        auto tensor_i = tensors_[i];
        if (tensor_i->dtype() != common_dtype_) {
            if (common_dtype_ == ScalarType::Undefined) {
                common_dtype_ = tensor_i->dtype();
            } else {
                common_dtype_ = update_common_dtype(common_dtype_, tensor_i->dtype());
            }
        }
    }
}

void TensorIterator::allocate_reduction_output_if_need() {
    if (is_reduction_) {
        auto device = common_device_;
        auto dtype = common_dtype_;
        auto input_shape = tensors_[num_outputs_]->shape();
        for (int i = 0; i < num_outputs_; ++i) {
            if (!tensors_[i]->defined()) {
                int64_t shape[MAX_TENSOR_DIMS];
                for (int k = 0; k < ndim_; ++k) {
                    shape[k] = input_shape[k];
                }
                shape[reduce_dim_] = 1;
                *tensors_[i] = std::move(empty(shape, ndim_, dtype, device, false));
            }
        }
    }
}

void TensorIterator::mark_outputs() {
    for (int i = 0; i < num_outputs_; ++i) {
        tensor_props_[i].is_output = true;
        auto output = tensors_[i];
        if (!output->defined()) continue;
        for (int arg = num_outputs_; arg < num_tensors_; ++arg) {
            auto input = tensors_[arg];
            if (output == input) {
                tensor_props_[i].is_read_write = true;
            }
        }
    }
}

void TensorIterator::check_mem_overlaps() {
    if (!check_mem_overlap_) return;
    for (int i = 0; i < num_outputs_; ++i) {
        auto output = tensors_[i];
        if (!output->defined()) continue;
        CHECK_FAIL(is_non_overlapping_and_dense(output->shape(), output->stride(), ndim_));
        for (int j = num_outputs_; j < num_tensors_; ++j) {
            auto input = tensors_[j];
            if (output != input) {
                CHECK_FAIL(is_no_partial_overlap(
                    output->data_ptr(), output->element_size_in_bytes(), output->shape(), output->stride(),
                    input->data_ptr(), input->element_size_in_bytes(), input->shape(), input->stride(),
                    ndim_));
            }
        }
    }
}

void TensorIterator::compute_broadcasted_shape() {
    for (int i = ndim_ - 1; i >= 0; --i) {
        bool is_first = true;
        int64_t sz;
        for (int j = 0; j < num_tensors_; ++j) {
            if (!tensors_[j]->defined()) continue;
            if (is_first) {
                sz = tensors_[j]->shape(i);
                is_first = false;
            } else {
                auto sz_ = tensors_[j]->shape(i);
                CHECK_FAIL(sz == sz_ || sz == 1 || sz_ == 1);
                sz = sz == 1 ? sz_ : sz;
            }
        }
        shape_[i] = sz;
    }
}

void TensorIterator::mark_resize_outputs() {
    // Outputs cannot be broadcasted. Check that the shape of the outputs matches
    // the inferred shape.
    for (int i = 0; i < num_outputs_; ++i) {
        auto output = tensors_[i];
        if (!output->defined()) {
            tensor_props_[i].will_resize = true;
        }
        if (output->defined() && !output->shape().equals(shape_)) {
            if (resize_outputs_ && !tensor_props_[i].is_read_write) {
                tensor_props_[i].will_resize = true;
                continue;
            }
            // for reduction, output size does not match shape_, as output is reduced size, and shape_ is size of the input
            CHECK_FAIL(is_reduction_, "output with shape ", output->shape(), " doesn't match the broadcast shape ",
                       this->shape_vec());
        }
    }
}

void TensorIterator::compute_broadcasted_strides() {
    for (int id = 0; id < num_tensors_; ++id) {
        auto t = tensors_[id];
        if (!t->defined()) continue;
        auto element_size_in_bytes = t->element_size_in_bytes();
        for (int i = ndim_ - 1; i >= 0; --i) {
            if (t->shape(i) == 1 && shape_[i] != 1) {
                stride_bytes_[id][i] = 0;
            } else {
                stride_bytes_[id][i] = t->stride(i) * element_size_in_bytes;
            }
        }
    }
}

void TensorIterator::permute_dimensions() {
    int64_t shape_temp[MAX_TENSOR_DIMS];
    int64_t strides_temp[MAX_TENSORS][MAX_TENSOR_DIMS];
    for (int i = 0; i < ndim_; ++i)
        shape_temp[i] = shape_[i];
    for (int i = 0; i < num_tensors_; ++i)
        for (int j = 0; j < ndim_; ++j)
            strides_temp[i][j] = stride_bytes_[i][j];
    for (int i = 0; i < ndim_; ++i)
        shape_[i] = shape_temp[perm_[i]];
    for (int i = 0; i < num_tensors_; ++i) {
        if (!tensors_[i]->defined()) continue;
        for (int j = 0; j < ndim_; ++j)
            stride_bytes_[i][j] = strides_temp[i][perm_[j]];
    }
}

void TensorIterator::reorder_dimensions() {
    if (ndim_ == 1) {
        perm_[0] = 0;
        return;
    }

    // initialize perm with n-1, n-2, ..., 1, 0
    for (int i = ndim_ - 1, ct = 0; i >= 0; --i) {
        perm_[ct++] = i;
    }

    // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
    // before dim1, and 0 if the comparison is ambiguous.
    auto should_swap = [&](size_t dim0, size_t dim1) {
        for (int arg = 0; arg < num_tensors_; ++arg) {
            // ignore undefined or incorrectly sized tensors
            if (!tensors_[arg]->defined() || tensor_props_[arg].will_resize) {
                continue;
            }
            int64_t stride0 = stride_bytes_[arg][dim0];
            int64_t stride1 = stride_bytes_[arg][dim1];
            if (is_reduction_ && arg < num_outputs_) {
                // move reduced dimensions to the front
                // strides of reduced dimensions are always set to 0 by review_reduce_result
                if ((stride0 == 0) != (stride1 == 0)) {
                    return stride1 == 0 ? 1 : -1;
                }
            }
            if (stride0 == 0 || stride1 == 0) {
                // move on to the next input if one of the dimensions is broadcasted
                continue;
            } else if (stride0 < stride1) {
                return -1;
            } else if (stride0 > stride1) {
                return 1;
            } else {
                // for equal strides, the dimension with smaller size goes front
                auto t_dim0 = shape_[dim0];
                auto t_dim1 = shape_[dim1];
                // return only if dimensions should be swapped, otherwise move on to the next tensor
                if (t_dim0 > t_dim1) {
                    return 1;
                }
            }
        }
        return 0;
    };

    // insertion sort with support for ambiguous comparisons
    for (int i = 1; i < ndim_; ++i) {
        int dim1 = i;
        for (int dim0 = i - 1; dim0 >= 0; dim0--) {
            int comparison = should_swap(perm_[dim0], perm_[dim1]);
            if (comparison > 0) {
                std::swap(perm_[dim0], perm_[dim1]);
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
        }
    }

    permute_dimensions();
}

void TensorIterator::allocate_outputs() {
    auto device = common_device_;
    auto dtype = common_dtype_;
    for (int i = 0; i < num_outputs_; ++i) {
        if (!tensors_[i]->defined() || tensor_props_[i].will_resize) {
            int64_t shape[MAX_TENSOR_DIMS];
            for (int k = 0; k < ndim_; ++k)
                shape[perm_[k]] = shape_[k];
            *tensors_[i] = std::move(empty(shape, ndim_, dtype, device, false));
            auto &stride = tensors_[i]->stride();
            for (int d = 0; d < ndim_; ++d) {
                stride_bytes_[i][d] = stride[perm_[d]] * element_size(dtype);
            }
        }
    }
}

void TensorIterator::coalesce_dimensions() {
    if (ndim_ <= 1) return;
    // We can coalesce two adjacent dimensions if either dim has size 1 or if:
    // shape[n] * stride[n] == stride[n + 1].
    auto can_coalesce = [&](int dim0, int dim1) {
        auto shape0 = shape_[dim0];
        auto shape1 = shape_[dim1];
        if (shape0 == 1 || shape1 == 1) {
            return true;
        }
        for (int i = 0; i < num_tensors_; ++i) {
            auto stride0 = stride_bytes_[i][dim0];
            auto stride1 = stride_bytes_[i][dim1];
            if (shape0 * stride0 != stride1) {
                return false;
            }
        }
        return true;
    };

    // replace each operands stride at dim0 with its stride at dim1
    auto replace_stride = [&](int dim0, int dim1) {
        for (int i = 0; i < num_tensors_; ++i) {
            stride_bytes_[i][dim0] = stride_bytes_[i][dim1];
        }
    };

    int prev_dim = 0;
    for (int dim = 1; dim < ndim_; ++dim) {
        if (can_coalesce(prev_dim, dim)) {
            if (shape_[prev_dim] == 1) {
                replace_stride(prev_dim, dim);
            }
            shape_[prev_dim] *= shape_[dim];
        } else {
            prev_dim++;
            if (prev_dim != dim) {
                replace_stride(prev_dim, dim);
                shape_[prev_dim] = shape_[dim];
            }
        }
    }

    ndim_ = prev_dim + 1;
}

void TensorIterator::update_data_pointers() {
    for (int i = 0; i < MAX_TENSORS; i++) {
        if (i < num_tensors_) {
            data_ptr_[i] = tensors_[i]->data_ptr();
        } else {
            data_ptr_[i] = nullptr;
        }
    }
}

std::ostream &operator<<(std::ostream &os, const TensorIterator &iter) {
    os << "TensorIterator(\n\tshape=[";
    for (int i = 0; i < iter.dim(); ++i)
        os << iter.shape(i) << ",";
    os << "\b],\n\t";
    for (int i = 0; i < iter.ntensors(); ++i) {
        os << "stride_bytes_" << i << "=[";
        for (int j = 0; j < iter.dim(); ++j)
            os << iter.stride_bytes(i, j) << ",";
        os << "\b],\n\t";
    }
    for (int i = 0; i < iter.ntensors(); ++i) {
        os << "data_ptr_" << i << "=" << iter.data_ptr(i) << ", \n\t";
    }
    os << "perm=[";
    for (int i = 0; i < iter.dim(); ++i)
        os << iter.perm(i) << ",";
    os << "\b],\n\tdim=" << iter.dim() << ",\n\tninputs=" << iter.ninputs();
    os << ",\n\tnoutputs=" << iter.noutputs();
    os << ",\n\tcommon_dtype=" << iter.common_dtype() << ")";
    return os;
}

TensorIterator::TensorIterator() {
    for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
        view_offsets_[i] = 0;
        shape_[i] = 0;
    }
}

TensorIterator &TensorIterator::add_output(Tensor &output) {
    CHECK_FAIL(num_inputs_ == 0);
    tensors_[num_tensors_++] = &output;
    num_outputs_++;
    return *this;
}

TensorIterator &TensorIterator::add_input(const Tensor &input) {
    tensors_[num_tensors_++] = const_cast<Tensor *>(&input);
    num_inputs_++;
    return *this;
}

int64_t TensorIterator::numel() const {
    int64_t numel = 1;
    for (int i = 0; i < ndim_; ++i) {
        numel *= shape_[i];
    }
    return numel;
}

int64_t TensorIterator::num_output_elements() const {
    int64_t elem = 1;
    for (int dim = 0; dim < ndim(); dim++) {
        if (stride_bytes_[0][dim] != 0 || shape_[dim] == 0) {
            elem *= shape_[dim];
        }
    }
    return elem;
}

int TensorIterator::num_reduce_dims() const {
    int count = 0;
    for (int dim = 0; dim < ndim(); dim++) {
        if (stride_bytes_[0][dim] == 0) {
            count++;
        }
    }
    return count;
}

bool TensorIterator::can_use_32bit_indexing() const {
    int64_t max_value = std::numeric_limits<int32_t>::max();
    if (numel() > max_value) {
        return false;
    }
    for (int i = 0; i < num_tensors_; ++i) {
        int64_t max_offset = 1;
        for (int d = 0; d < ndim(); ++d) {
            max_offset += (shape_[d] - 1) * stride_bytes_[i][d];
        }
        if (max_offset > max_value) {
            return false;
        }
    }
    return true;
}

bool TensorIterator::has_contiguous_first_dim() const {
    for (int i = 0; i < num_tensors_; ++i) {
        if (stride_bytes_[i][0] != tensors_[i]->element_size_in_bytes()) {
            return false;
        }
    }
    return true;
}

bool TensorIterator::is_contiguous() const {
    if (numel() == 1) {
        return true;
    }
    if (ndim() != 1) {
        return false;
    }
    return has_contiguous_first_dim();
}

int TensorIterator::get_dim_to_split() const {
    CHECK_FAIL(ndim() >= 1);
    int64_t max_extent = -1;
    int dim_to_split = -1;
    for (int dim = ndim() - 1; dim >= 0; dim--) {
        const int64_t size = shape_[dim];
        if (size == 0) {
            continue;
        }
        for (int i = 0; i < num_tensors_; ++i) {
            // std::abs is necessary to handle some special cases where we support negative strides
            const int64_t extent = (size - 1) * std::abs(stride_bytes_[i][dim]);
            if (extent > max_extent) {
                max_extent = extent;
                dim_to_split = dim;
            }
        }
    }
    CHECK_FAIL(max_extent >= 0);
    return dim_to_split;
}

bool TensorIterator::is_dim_reduced(int dim) const {
    for (int i = 0; i < num_tensors_; ++i) {
        if ((i < num_outputs_) && stride_bytes_[i][dim] == 0 && shape_[dim] > 1) {
            return true;
        }
    }
    return false;
}

void TensorIterator::narrow(int dim, int64_t start, int64_t size) {
    CHECK_FAIL(dim < ndim() && size >= 1);
    shape_[dim] = size;
    view_offsets_[dim] += start;
    for (int i = 0; i < num_tensors_; ++i) {
        data_ptr_[i] = (char *)data_ptr_[i] + stride_bytes_[i][dim] * start;
    }
    if (size == 1 && !is_reduction_) {
        coalesce_dimensions();
    }
}

std::unique_ptr<TensorIterator> TensorIterator::split(int dim) {
    CHECK_FAIL(dim >= 0 && dim < ndim() && shape()[dim] >= 2);
    auto copy = std::make_unique<TensorIterator>(*this);
    bool overlaps = is_dim_reduced(dim);
    auto copy_size = shape_[dim] / 2;
    auto this_size = shape_[dim] - copy_size;
    copy->narrow(dim, 0, copy_size);
    copy->final_output_ &= !overlaps;
    this->narrow(dim, copy_size, this_size);
    this->accumulate_ |= overlaps;
    return copy;
}

SplitUntil32Bit TensorIterator::with_32bit_indexing() const {
    return SplitUntil32Bit(*this);
}

TensorIterator &TensorIterator::build() {
    // check whether all defined tensors are in the same device, then update common_device_
    check_and_compute_common_device();
    // check whether all defined tensors have the same dim, then update ndim_
    check_and_compute_dim();
    // infer common_dtype from input tensors
    compute_common_dtype();
    // atomatic output allocation for reduction
    allocate_reduction_output_if_need();
    // set is_output and is_read_write flags on appropriate tensors
    mark_outputs();
    // check that the defined outputs have no internal overlap
    // and do not share memory with inputs
    check_mem_overlaps();
    // compute and check the broadcasted shape through input tensors
    compute_broadcasted_shape();
    // mark outputs for resizing if necessary
    mark_resize_outputs();
    // compute each defined tensor's stride after broadcasting
    compute_broadcasted_strides();
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_outputs();
    // coalesce adjacent dimensions when possible
    coalesce_dimensions();
    // update data_ptr_
    update_data_pointers();
    return *this;
}

TensorIterator &TensorIterator::build_for_loops() {
    is_reduction_ = false;
    resize_outputs_ = true;
    return this->build();
}

TensorIterator &TensorIterator::build_for_reduce(int64_t reduce_dim) {
    is_reduction_ = true;
    resize_outputs_ = false;
    reduce_dim_ = reduce_dim;
    return this->build();
}
