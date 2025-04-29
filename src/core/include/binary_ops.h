#pragma once

#include "tensor.h"

namespace gpu {

Tensor &add_out(Tensor &out, const Tensor &left, const Tensor &right);
Tensor add(const Tensor &left, const Tensor &right);
Tensor &add_(Tensor &self, const Tensor &other);
Tensor &sub_out(Tensor &out, const Tensor &left, const Tensor &right);
Tensor sub(const Tensor &left, const Tensor &right);
Tensor &sub_(Tensor &self, const Tensor &other);
Tensor &mul_out(Tensor &out, const Tensor &left, const Tensor &right);
Tensor mul(const Tensor &left, const Tensor &right);
Tensor &mul_(Tensor &self, const Tensor &other);
Tensor &div_out(Tensor &out, const Tensor &left, const Tensor &right);
Tensor div(const Tensor &left, const Tensor &right);
Tensor &div_(Tensor &self, const Tensor &other);

} // namespace gpu
