#pragma once

#include "tensor_iterator.h"

void add_kernel(TensorIterator &iter);
void sub_kernel(TensorIterator &iter);
void mul_kernel(TensorIterator &iter);
void div_kernel(TensorIterator &iter);
