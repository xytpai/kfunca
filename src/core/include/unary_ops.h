#pragma once

#include "tensor.h"

namespace gpu {

Tensor clone(const Tensor &self);
Tensor convert(const Tensor &self, ScalarType dtype);

} // namespace gpu
