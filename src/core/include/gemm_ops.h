#pragma once

#include "tensor.h"

namespace gpu {

void gemm_out(Tensor &out, const Tensor &a, const Tensor &b, float alpha, float beta);
Tensor gemm(const Tensor &a, const Tensor &b, float alpha, float beta);

} // namespace gpu
