#pragma once

#include "tensor_iterator.h"

void index_put_kernel(TensorIterator &iter, const std::vector<int64_t> index_size, const std::vector<int64_t> index_stride);
