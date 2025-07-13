#include "tensor_iterator.h"
#include "unary_ops_kernel.h"

namespace gpu {

Tensor convert(const Tensor &self, ScalarType dtype) {
    Tensor out = empty(self.sizes(), dtype, self.device());
    auto iter = TensorIterator().add_output(out).add_input(self).build_for_loops();
    copy_kernel(iter);
    return out;
}

} // namespace gpu
