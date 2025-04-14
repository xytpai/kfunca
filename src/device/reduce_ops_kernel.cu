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

template <typename scalar_t, typename acc_t = scalar_t, typename factor_t = acc_t, typename out_t = acc_t>
struct MeanOps {
    factor_t factor;

    DEVICE acc_t reduce(acc_t a, scalar_t b, int64_t /*idx*/) const {
        return combine(a, static_cast<acc_t>(b));
    }

    DEVICE_INLINE acc_t combine(acc_t a, acc_t b) const {
        return a + b;
    }

    DEVICE_INLINE out_t project(acc_t a) const {
        return a * factor;
    }

    static DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
        return acc;
    }

    DEVICE_INLINE acc_t gpu_shfl_down(ITEM &item, acc_t data, int offset) const {
        return GPU_SHFL_DOWN(item, data, offset);
    }

    MeanOps(factor_t factor) :
        factor(factor) {
    }
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
void mean_kernel_impl(TensorIterator &iter) {
    acc_t factor = static_cast<acc_t>(iter.num_output_elements()) / iter.numel();
    gpu_reduce_kernel<scalar_t, out_t>(iter, MeanOps<scalar_t, acc_t, acc_t, out_t>{factor});
}

void mean_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.dtype(), "mean_kernel", [&]() {
        mean_kernel_impl<scalar_t>(iter);
    });
}
