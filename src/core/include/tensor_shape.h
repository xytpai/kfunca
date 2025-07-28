#pragma once

#include <vector>

#include "tensor_iterator.h"

namespace gpu {

Tensor concat(const std::vector<Tensor> tensors, int64_t dim);
std::vector<Tensor> tensor_split(const Tensor &self, std::vector<int64_t> indices, int64_t dim);

} // namespace gpu
