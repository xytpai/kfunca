#pragma once

#include "tensor.h"

namespace gpu {

Tensor sum(const Tensor &self, int64_t reduce_dim);

} // namespace gpu
