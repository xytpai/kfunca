#pragma once

#include "tensor.h"

namespace gpu {

std::tuple<Tensor, Tensor> sort(
    const Tensor &self,
    int64_t dim,
    bool descending);

std::tuple<Tensor, Tensor> topk(
    const Tensor &self,
    int64_t k,
    int64_t dim,
    bool largest);

} // namespace gpu
