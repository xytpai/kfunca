#include "tensor_iterator.h"
#include "nullary_ops_kernel.h"

namespace gpu {

Tensor &fill_out(Tensor &out, const any_t &value) {
    auto iter = TensorIterator().add_output(out).resize_outputs(false).build();
    fill_kernel(iter, value);
    return out;
}

Tensor &fill_(Tensor &self, const any_t &value) {
    return fill_out(self, value);
}

} // namespace gpu
