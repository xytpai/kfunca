#pragma once

#include "tensor_iterator.h"

std::tuple<Tensor, Tensor> sort_stable_kernel(
    const Tensor &self,
    int64_t dim,
    bool descending);
