#include "tensor_iterator.h"
#include "gemm_kernel.h"

namespace gpu {

void gemm_out(Tensor &out, const Tensor &a, const Tensor &b, float alpha, float beta) {
    gemm_kernel(out, a, b, alpha, beta);
}

Tensor gemm(const Tensor &a, const Tensor &b, float alpha, float beta) {
    auto out_size = a.sizes();
    out_size[out_size.size() - 1] = b.shape(-1);
    Tensor out = empty(out_size, a.dtype(), a.device());
    gemm_kernel(out, a, b, alpha, beta);
    return out;
}

} // namespace gpu
