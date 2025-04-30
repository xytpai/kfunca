#pragma once

#include "tensor_iterator.h"

void sum_kernel(TensorIterator &iter);
void mean_kernel(TensorIterator &iter);
void mean_var_kernel(TensorIterator &iter, double correction, bool take_sqrt);
