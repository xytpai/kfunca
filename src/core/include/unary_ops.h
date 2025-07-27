#pragma once

#include "tensor.h"

namespace gpu {

Tensor clone(const Tensor &self);
Tensor &copy_(Tensor &self, const Tensor &other);
Tensor convert(const Tensor &self, ScalarType dtype);

} // namespace gpu
