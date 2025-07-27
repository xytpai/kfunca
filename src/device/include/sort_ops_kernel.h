#pragma once

#include "tensor_iterator.h"

std::tuple<Tensor, Tensor> sort_stable_kernel(
    const Tensor &self,
    int64_t dim,
    bool descending);

std::tuple<Tensor, Tensor> topk_with_sort(
    const Tensor &self,
    int64_t k,
    int64_t dim,
    bool largest);
