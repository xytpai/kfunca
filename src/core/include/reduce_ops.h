#pragma once

#include "tensor.h"

namespace gpu {

Tensor sum(const Tensor &self, int64_t reduce_dim);
Tensor mean(const Tensor &self, int64_t reduce_dim);
std::tuple<Tensor, Tensor> mean_var(const Tensor &self, int64_t reduce_dim, bool take_sqrt);

} // namespace gpu
