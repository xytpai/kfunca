#pragma once

#include <vector>

#include "tensor_iterator.h"

namespace gpu {

Tensor &index_put_(Tensor &self, const std::vector<Tensor> &indices, const Tensor &values);

} // namespace gpu
