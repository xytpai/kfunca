#include "tensor_iterator.h"
#include "binary_ops_kernel.h"

Tensor add(Tensor &left, Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    add_kernel(iter);
    return out;
}
