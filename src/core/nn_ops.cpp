#include "tensor_iterator.h"
#include "causal_attention_kernel.h"

namespace gpu {

Tensor causal_attention(const Tensor &q, const Tensor &k, const Tensor &v) {
    return causal_attention_kernel(q, k, v);
}

} // namespace gpu
