#include <limits>
#include <algorithm>

#include "launcher.h"
#include "scalar_type.h"
#include "tensor_iterator.h"

void gemm_kernel(Tensor &out, const Tensor &a, const Tensor &b, float alpha, float beta) {
    CHECK_FAIL(out.is_contiguous() && a.is_contiguous() && b.is_contiguous());
    int m = 1;
    for (auto &d : a.sizes()) {
        m *= d;
    }
    int k = a.shape(-1);
    m /= k;
    CHECK_FAIL(b.dim() == 2 && b.shape(0) == k);
    CHECK_FAIL(a.dtype() == b.dtype());
    int n = b.shape(-1);
    CHECK_FAIL(out.shape(-1) == n);
    int m_ = 1;
    for (auto &d : out.sizes()) {
        m_ *= d;
    }
    m_ /= n;
    CHECK_FAIL(m == m_);
    DISPATCH_FLOATING_TYPES(a.dtype(), "gemm_kernel", [&]() {
        int status = gemm_ref<scalar_t, scalar_t, true, true, true>(m, n, k,
                                                                    alpha,
                                                                    a.data_ptr<scalar_t>(),
                                                                    k,
                                                                    b.data_ptr<scalar_t>(),
                                                                    n,
                                                                    beta,
                                                                    out.data_ptr<scalar_t>(),
                                                                    n);
        CHECK_FAIL(status == 0);
    });
}
