#pragma once

#include "tensor.h"

namespace gpu {

Tensor causal_attention(const Tensor &q, const Tensor &k, const Tensor &v);

} // namespace gpu
