#pragma once

#include <vector>

#include "tensor_iterator.h"

namespace gpu {

Tensor concat(const std::vector<Tensor> tensors, int64_t dim);

}
