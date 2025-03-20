#include "tensor_iterator.h"
#include "tensor_reduce.h"
#include "scalar_type.h"

template <typename scalar_t>
struct SumFunctor {
    DEVICE scalar_t operator()(scalar_t a, scalar_t b) const {
        return a + b;
    }
};

void sum_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.dtype(), "sum_kernel", [&]() {
        gpu_reduce_kernel<scalar_t, scalar_t>(iter,
                                              func_wrapper<scalar_t>(SumFunctor<scalar_t>()));
    });
}
