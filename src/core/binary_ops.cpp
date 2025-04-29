#include "tensor_iterator.h"
#include "binary_ops_kernel.h"

namespace gpu {

Tensor &add_out(Tensor &out, const Tensor &left, const Tensor &right) {
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    add_kernel(iter);
    return out;
}

Tensor add(const Tensor &left, const Tensor &right) {
    Tensor out;
    out = add_out(out, left, right);
    return out;
}

Tensor &add_(Tensor &self, const Tensor &other) {
    return add_out(self, self, other);
}

Tensor &sub_out(Tensor &out, const Tensor &left, const Tensor &right) {
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    sub_kernel(iter);
    return out;
}

Tensor sub(const Tensor &left, const Tensor &right) {
    Tensor out;
    out = sub_out(out, left, right);
    return out;
}

Tensor &sub_(Tensor &self, const Tensor &other) {
    return sub_out(self, self, other);
}

Tensor &mul_out(Tensor &out, const Tensor &left, const Tensor &right) {
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    mul_kernel(iter);
    return out;
}

Tensor mul(const Tensor &left, const Tensor &right) {
    Tensor out;
    out = mul_out(out, left, right);
    return out;
}

Tensor &mul_(Tensor &self, const Tensor &other) {
    return mul_out(self, self, other);
}

Tensor &div_out(Tensor &out, const Tensor &left, const Tensor &right) {
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    div_kernel(iter);
    return out;
}

Tensor div(const Tensor &left, const Tensor &right) {
    Tensor out;
    out = div_out(out, left, right);
    return out;
}

Tensor &div_(Tensor &self, const Tensor &other) {
    return div_out(self, self, other);
}

} // namespace gpu
