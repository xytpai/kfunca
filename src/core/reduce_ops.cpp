#include "tensor_iterator.h"
#include "reduce_ops_kernel.h"

#include <tuple>

namespace gpu {

Tensor sum(const Tensor &self, int64_t reduce_dim) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(self).build_for_reduce(reduce_dim);
    sum_kernel(iter);
    return out;
}

Tensor mean(const Tensor &self, int64_t reduce_dim) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(self).build_for_reduce(reduce_dim);
    mean_kernel(iter);
    return out;
}

std::tuple<Tensor, Tensor> mean_var(const Tensor &self, int64_t reduce_dim, bool take_sqrt) {
    Tensor mean, var;
    auto iter = TensorIterator().add_output(var).add_output(mean).add_input(self).build_for_reduce(reduce_dim);
    double correction = 1;
    mean_var_kernel(iter, correction, take_sqrt);
    return std::make_tuple(mean, var);
}

} // namespace gpu
