#include "tensor_shape.h"

namespace gpu {

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
inline void check_cat_shape_except_dim(
    const Tensor &first,
    const Tensor &second,
    int64_t dimension,
    int64_t index) {
    CHECK_FAIL(first.device() == second.device());
    int64_t first_dims = first.dim();
    int64_t second_dims = second.dim();
    CHECK_FAIL(
        first_dims == second_dims,
        "Tensors must have same number of dimensions: got ",
        first_dims,
        " and ",
        second_dims);
    for (int64_t dim = 0; dim < first_dims; dim++) {
        if (dim == dimension) {
            continue;
        }
        int64_t first_dim_size = first.shape(dim);
        int64_t second_dim_size = second.shape(dim);
        CHECK_FAIL(
            first_dim_size == second_dim_size,
            "Sizes of tensors must match except in dimension ",
            dimension,
            ". Expected size ",
            static_cast<long long>(first_dim_size),
            " but got size ",
            static_cast<long long>(second_dim_size),
            " for tensor number ",
            index,
            " in the list.");
    }
}

Tensor concat(const std::vector<Tensor> tensors, int64_t dim) {
    dim = maybe_wrap_dim(dim, tensors[0].dim());
    auto out_dtype = tensors[0].dtype();

    // valid check
    bool all_contiguous = true;
    bool all_same_dtype = true;
    size_t size_at_dim = tensors[0].shape(dim);

    for (int i = 1; i < tensors.size(); ++i) {
        const Tensor &t = tensors[i];
        all_same_dtype = all_same_dtype && out_dtype == t.dtype();
        check_cat_shape_except_dim(tensors[0], t, dim, i);
        size_at_dim += t.shape(dim);
        all_contiguous = all_contiguous && t.is_contiguous();
    }

    auto out_size = tensors[0].sizes();
    out_size[dim] = size_at_dim;
    Tensor result = empty(out_size, out_dtype, tensors[0].device());

    int64_t offset = 0;
    for (const Tensor &t : tensors) {
        int64_t dim_size = t.shape(dim);
        Tensor nt = result.narrow(dim, offset, dim_size);
        nt.copy_(t);
        offset += dim_size;
    }
    return result;
}

std::vector<Tensor> tensor_split(const Tensor &self, std::vector<int64_t> indices, int64_t dim) {
    CHECK_FAIL(
        self.dim() > 0,
        "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
        self.dim(),
        " dims");
    int64_t dim_ = maybe_wrap_dim(dim, self.dim());
    int64_t num_indices = indices.size();
    std::vector<Tensor> splits(num_indices);
    int64_t start_idx = 0;
    for (int split_idx = 0; split_idx < num_indices; ++split_idx) {
        auto end_idx = start_idx + indices[split_idx];
        splits[split_idx] = self.slice(dim_, start_idx, end_idx);
        start_idx = end_idx;
    }
    CHECK_FAIL(start_idx == self.shape(dim));
    return splits;
}

} // namespace gpu
