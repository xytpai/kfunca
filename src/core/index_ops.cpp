#include "tensor_iterator.h"
#include "index_ops_kernel.h"

namespace gpu {

Tensor &index_put_(Tensor &self, const std::vector<Tensor> &indices, const Tensor &values) {
    CHECK_FAIL(indices.size() == self.dim(),
               "Number of indices must match the number of dimensions in the tensor.");
    CHECK_FAIL(self.defined() && values.defined(),
               "Both self and values tensors must be defined.");
    CHECK_FAIL(self.dtype() == values.dtype(),
               "Data types of self and values tensors must match.");

    int64_t element_size_bytes = element_size(self.dtype());
    std::vector<int64_t> indexed_sizes, indexed_strides;
    for (int dim = 0; dim < indices.size(); ++dim) {
        indexed_sizes.push_back(self.shape(dim));
        indexed_strides.push_back(self.stride(dim) * element_size_bytes);
    }
    auto replacement_shape = indices[0].sizes();
    std::vector<int64_t> replacement_strides(replacement_shape.size(), 0);
    auto self_ = self.as_strided(replacement_shape, replacement_strides);

    if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end()) {
        CHECK_FAIL(
            false, "index is out of bounds for dimension with size 0");
    }

    auto iter = TensorIterator().add_output(self_).check_mem_overlap(false).resize_outputs(false);
    iter = iter.add_input(values);
    for (auto &index : indices) {
        CHECK_FAIL(index.defined() && index.dtype() == ScalarType::Long, "Indices must be of type Long.");
        iter = iter.add_input(index);
    }
    iter = iter.build();
    index_put_kernel(iter, indexed_sizes, indexed_strides);
    return self;
}

} // namespace gpu
