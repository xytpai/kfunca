#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"
#include "accumulate_type.h"

#include "causal_attention.h"
#include "causal_attention_ref.h"

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
    auto qk_temp = empty({batch_size, nheads, q_seq_length, kv_seq_length}, out.dtype(), out.device());
    auto out_m = empty({batch_size, nheads, q_seq_length}, out.dtype(), out.device());
    auto out_l = empty({batch_size, nheads, q_seq_length}, out.dtype(), out.device());
    DISPATCH_NN_TYPES(q.dtype(), "causal_attention_kernel", [&]() {
        int ret = 1;
        switch (hidden_size) {
        case 128:
            ret = causal_attention_forward_fma_ref<scalar_t, 128>(out.data_ptr<scalar_t>(),
                                                                  q.data_ptr<scalar_t>(),
                                                                  k.data_ptr<scalar_t>(),
                                                                  v.data_ptr<scalar_t>(),
                                                                  batch_size,
                                                                  nheads,
                                                                  q_seq_length,
                                                                  kv_seq_length,
                                                                  out_m.data_ptr<scalar_t>(),
                                                                  out_l.data_ptr<scalar_t>());
            break;
        case 64:
            ret = causal_attention_forward_fma_ref<scalar_t, 64>(out.data_ptr<scalar_t>(),
                                                                 q.data_ptr<scalar_t>(),
                                                                 k.data_ptr<scalar_t>(),
                                                                 v.data_ptr<scalar_t>(),
                                                                 batch_size,
                                                                 nheads,
                                                                 q_seq_length,
                                                                 kv_seq_length,
                                                                 out_m.data_ptr<scalar_t>(),
                                                                 out_l.data_ptr<scalar_t>());
            break;
        default:
            break;
        }
        if (ret != 0) {
            causal_attention_ref_forward<scalar_t>(
                out.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                batch_size,
                nheads,
                q_seq_length,
                kv_seq_length,
                hidden_size,
                qk_temp.data_ptr<scalar_t>(),
                out_m.data_ptr<scalar_t>(),
                out_l.data_ptr<scalar_t>());
        }
    });
    return out;
}
