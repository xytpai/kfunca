#include "tensor_iterator.h"
#include "sort_ops_kernel.h"

namespace gpu {

std::tuple<Tensor, Tensor> sort(
    const Tensor &self,
    int64_t dim,
    bool descending) {
    return sort_stable_kernel(self, dim, descending);
}

} // namespace gpu
