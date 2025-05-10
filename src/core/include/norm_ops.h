#pragma once

#include "tensor.h"

namespace gpu {

std::tuple<Tensor, Tensor> norm_stat(const Tensor &self, const int dim);

} // namespace gpu
