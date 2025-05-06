#pragma once

#include "tensor.h"

namespace gpu {

Tensor &fill_out(Tensor &out, const any_t &value);
Tensor &fill_(Tensor &self, const any_t &value);

} // namespace gpu
