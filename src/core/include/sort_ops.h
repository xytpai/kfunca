#pragma once

#include "tensor.h"

namespace gpu {

std::tuple<Tensor, Tensor> sort(
    const Tensor &self,
    int64_t dim,
    bool descending);

} // namespace gpu
