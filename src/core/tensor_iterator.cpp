#include "tensor_iterator.h"

bool TensorIterator::check_and_compute_dim() {
    bool is_first = true;
    int first_dim;
    for (int i = 0; i < num_tensors_; ++i) {
        if (!tensors_[i]->defined()) continue;
        if (is_first) {
            first_dim = tensors_[i]->dim();
            is_first = false;
        } else {
            if (first_dim != tensors_[i]->dim())
                return false;
        }
    }
    ndim_ = first_dim;
    return true;
}

void TensorIterator::compute_shape() {
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

void TensorIterator::compute_types() {
    int common_device = -1;
    common_dtype_ = ScalarType::Undefined;
    ScalarType output_dtype = ScalarType::Undefined;
    for (int i = num_outputs_; i < num_tensors_; i++) {
        auto tensor_i = tensors_[i];
        if (common_device == -1 && tensor_i->device() >= 0) {
            common_device = tensor_i->device();
        } else {
            CHECK_FAIL(tensor_i->device() == common_device, "All input tensors should in the same device");
        }
        if (tensor_i->dtype() != common_dtype_) {
            if (common_dtype_ == ScalarType::Undefined) {
                common_dtype_ = tensor_i->dtype();
            } else {
                common_dtype_ = update_common_dtype(common_dtype_, tensor_i->dtype());
            }
        }
    }
}

void TensorIterator::compute_strides() {
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
    int ct = 0;
    for (int i = ndim_ - 1; i >= 0; --i) {
        perm_[ct++] = i;
    }

    auto should_swap = [&](size_t dim0, size_t dim1) {
        for (int arg = 0; arg < num_tensors_; ++arg) {
            if (!tensors_[arg]->defined()) continue;
            int64_t stride0 = stride_bytes_[arg][dim0];
            int64_t stride1 = stride_bytes_[arg][dim1];
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
    is_reordered_ = true;
}

void TensorIterator::allocate_outputs() {
    auto device = tensors_[num_outputs_]->device();
    auto dtype = common_dtype_;
    for (int i = 0; i < num_outputs_; ++i) {
        if (!tensors_[i]->defined()) {
            if (!is_reordered_) {
                *tensors_[i] = std::move(empty(shape_, ndim_, dtype, device, false));
            } else {
                int64_t shape[MAX_TENSOR_DIMS];
                for (int k = 0; k < ndim_; ++k)
                    shape[perm_[k]] = shape_[k];
                *tensors_[i] = std::move(empty(shape, ndim_, dtype, device, false));
            }
            auto &stride = tensors_[i]->stride();
            for (int d = 0; d < ndim_; ++d) {
                stride_bytes_[i][d] = stride[ndim_ - 1 - d] * element_size(dtype);
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
