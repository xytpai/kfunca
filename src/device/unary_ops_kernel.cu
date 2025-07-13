#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"
#include "accumulate_type.h"

template <typename scalar_t>
struct CopyFunctor {
    DEVICE scalar_t operator()(scalar_t a) const {
        return a;
    }
};

void copy_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.dtype(), "copy_kernel", [&]() {
        gpu_kernel(iter, CopyFunctor<scalar_t>());
    });
}
