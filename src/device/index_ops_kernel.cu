#include "tensor_index.h"

template <typename scalar_t>
struct IndexOffsetFunctor {
    DEVICE void operator()(char *out_data, const char *in_data, int64_t offset) const {
        *reinterpret_cast<scalar_t *>(out_data + offset) = *reinterpret_cast<const scalar_t *>(in_data);
    }
};

void index_put_kernel(TensorIterator &iter, const std::vector<int64_t> index_size, const std::vector<int64_t> index_stride) {
    DISPATCH_BASIC_TYPES(iter.dtype(), "index_put_kernel", [&]() {
        auto offset_fn = IndexOffsetFunctor<scalar_t>();
        gpu_index_kernel(iter, index_size, index_stride, offset_fn);
    });
}
