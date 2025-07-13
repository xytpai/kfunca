#pragma once

#include "tensor.h"

namespace gpu {

Tensor convert(const Tensor &self, ScalarType dtype);

} // namespace gpu
