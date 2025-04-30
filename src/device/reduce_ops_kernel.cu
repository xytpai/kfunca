#include "tensor_iterator.h"
#include "tensor_reduce.h"
#include "scalar_type.h"
#include "accumulate_type.h"

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

template <typename scalar_t, typename index_t>
struct WelfordData {
    scalar_t mean;
    scalar_t m2;
    index_t n;
    scalar_t nf;
    HOST_DEVICE WelfordData() :
        mean(0), m2(0), n(0), nf(0) {
    }
    HOST_DEVICE WelfordData(
        scalar_t mean,
        scalar_t m2,
        index_t n,
        scalar_t nf) :
        mean(mean),
        m2(m2), n(n), nf(nf) {
    }
};

template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct WelfordOps {
    acc_scalar_t correction;
    bool take_sqrt;

public:
    using acc_t = WelfordData<acc_scalar_t, index_t>;
    inline DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
        // We accumulate n in index_t to avoid cumulative rounding error, but still
        // need nf for use in combine where int32 may overflow.
        index_t new_n = acc.n + 1;
        acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
        acc_scalar_t delta = data - acc.mean;
        acc_scalar_t new_mean = acc.mean + delta / new_nf;
        acc_scalar_t new_delta = data - new_mean;
        return {
            new_mean,
            acc.m2 + delta * new_delta,
            new_n,
            new_nf,
        };
    }
    inline DEVICE acc_t combine(acc_t a, acc_t b) const {
        if (a.nf == 0) {
            return b;
        }
        if (b.nf == 0) {
            return a;
        }
        acc_scalar_t delta = b.mean - a.mean;
        acc_scalar_t new_count = a.nf + b.nf;
        acc_scalar_t nb_over_n = b.nf / new_count;
        return {
            a.mean + delta * nb_over_n,
            a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
            // setting acc.n as -1 since acc.n might not be able to represent the count
            // correctly within its range, setting it to -1 to avoid confusion
            -1,
            new_count};
    }
    inline DEVICE res_t project(acc_t acc) const {
        const auto mean = static_cast<scalar_t>(acc.mean);
        const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
        const auto var = acc.m2 / divisor;
        res_t results(take_sqrt ? std::sqrt(var) : var, mean);
        return results;
    }
    static DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
        return acc;
    }
    DEVICE_INLINE acc_t gpu_shfl_down(ITEM &item, acc_t acc, int offset) const {
        return {
            GPU_SHFL_DOWN(item, acc.mean, offset), GPU_SHFL_DOWN(item, acc.m2, offset), GPU_SHFL_DOWN(item, acc.n, offset), GPU_SHFL_DOWN(item, acc.nf, offset)};
    }
    HOST_DEVICE WelfordOps(acc_scalar_t correction, bool take_sqrt) :
        correction(correction), take_sqrt(take_sqrt) {
    }
};

template <typename scalar_t, typename out_t = scalar_t>
void mean_var_kernel_impl(TensorIterator &iter, double correction, bool take_sqrt) {
    // reducing unrolling factor to 2 for welford kernel
    // This is necessary to lower register usage that leads to register spills.
    using accscalar_t = acc_type<scalar_t>;
    using ops_t = WelfordOps<scalar_t, accscalar_t, int32_t, std::pair<out_t, out_t>>;
    ops_t ops(static_cast<accscalar_t>(correction), take_sqrt);
    gpu_reduce_kernel<scalar_t, out_t, 2>(iter, ops, typename ops_t::acc_t{});
}

void mean_var_kernel(TensorIterator &iter, double correction, bool take_sqrt) {
    DISPATCH_FLOATING_TYPES(iter.dtype(), "mean_var_kernel", [&]() {
        mean_var_kernel_impl<scalar_t>(iter, correction, take_sqrt);
    });
}
