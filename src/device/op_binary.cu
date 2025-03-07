#include "tensor.h"
#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"

template <typename scalar_t>
struct AddFunctor {
    DEVICE scalar_t operator()(scalar_t a, scalar_t b) const {
        return a + b;
    }
};

Tensor add(Tensor &left, Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    DISPATCH_BASIC_TYPES(out.dtype(), "add", [&]() {
        gpu_kernel(iter, AddFunctor<scalar_t>());
    });
    return out;
}
