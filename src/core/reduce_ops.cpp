#include "tensor_iterator.h"
#include "reduce_ops_kernel.h"

namespace gpu {

Tensor sum(const Tensor &self, int64_t reduce_dim) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(self).build_for_reduce(reduce_dim);
    sum_kernel(iter);
    return out;
}

} // namespace gpu
