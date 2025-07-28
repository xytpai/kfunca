#include "unary_ops.h"
#include "tensor_iterator.h"
#include "unary_ops_kernel.h"

namespace gpu {

Tensor clone(const Tensor &self) {
    Tensor out = empty_like(self);
    out = copy_(out, self);
    return out;
}

Tensor &copy_(Tensor &self, const Tensor &other) {
    auto iter = TensorIterator().add_output(self).add_input(other).resize_outputs(false).check_mem_overlap(false).build_for_loops();
    copy_kernel(iter);
    return self;
}

Tensor convert(const Tensor &self, ScalarType dtype) {
    Tensor out = empty(self.sizes(), dtype, self.device());
    auto iter = TensorIterator().add_output(out).add_input(self).build_for_loops();
    copy_kernel(iter);
    return out;
}

} // namespace gpu
