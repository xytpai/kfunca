#pragma once

#include "tensor_iterator.h"

std::tuple<Tensor, Tensor> norm_stat_kernel(const Tensor &self, const int dim);
