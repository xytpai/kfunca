#include "tensor_iterator.h"
#include "norm_ops_kernel.h"

#include <tuple>

namespace gpu {

std::tuple<Tensor, Tensor> norm_stat(const Tensor &self, const int dim) {
    return norm_stat_kernel(self, dim);
}

} // namespace gpu
