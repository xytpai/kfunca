#pragma once

#include "tensor_iterator.h"

Tensor causal_attention_kernel(const Tensor &q, const Tensor &k, const Tensor &v);
