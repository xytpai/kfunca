#pragma once

#include "tensor.h"

namespace gpu {

Tensor add(const Tensor &left, const Tensor &right);
Tensor sub(const Tensor &left, const Tensor &right);
Tensor mul(const Tensor &left, const Tensor &right);
Tensor div(const Tensor &left, const Tensor &right);

} // namespace gpu
