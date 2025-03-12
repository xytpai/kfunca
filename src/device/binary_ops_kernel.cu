#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"

template <typename scalar_t>
struct AddFunctor {
    DEVICE scalar_t operator()(scalar_t a, scalar_t b) const {
        return a + b;
    }
};

void add_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "add_kernel", [&]() {
        gpu_kernel(iter, AddFunctor<scalar_t>());
    });
}
