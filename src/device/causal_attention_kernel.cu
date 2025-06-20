#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"
#include "accumulate_type.h"

#include "causal_attention.h"

Tensor causal_attention_kernel(const Tensor &q, const Tensor &k, const Tensor &v) {
    auto q_shape = q.sizes();
    auto k_shape = k.sizes();
    auto v_shape = v.sizes();
    int batch_size = q_shape[0];
    int nheads = q_shape[1];
    int q_seq_length = q_shape[2];
    int hidden_size = q_shape[3];
    int kv_seq_length = k_shape[2];
    CHECK_FAIL(k_shape[0] == batch_size && k_shape[1] == nheads && k_shape[3] == hidden_size);
    CHECK_FAIL(k_shape == v_shape);
    CHECK_FAIL(q.dtype() == k.dtype() && q.dtype() == v.dtype());
    auto out = empty_like(q);
    DISPATCH_NN_TYPES(q.dtype(), "causal_attention_kernel", [&]() {
    });
    return out;
}
