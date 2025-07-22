#pragma once

#include "tensor_iterator.h"

void gemm_kernel(Tensor &out, const Tensor &a, const Tensor &b, float alpha, float beta);
