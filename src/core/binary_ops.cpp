#include "tensor_iterator.h"
#include "binary_ops_kernel.h"

namespace gpu {

Tensor add(const Tensor &left, const Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    add_kernel(iter);
    return out;
}

Tensor sub(const Tensor &left, const Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    sub_kernel(iter);
    return out;
}

Tensor mul(const Tensor &left, const Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    mul_kernel(iter);
    return out;
}

Tensor div(const Tensor &left, const Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    div_kernel(iter);
    return out;
}

} // namespace gpu
