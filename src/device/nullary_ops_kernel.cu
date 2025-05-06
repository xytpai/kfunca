#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"
#include "accumulate_type.h"

template <typename scalar_t>
struct FillFunctor {
    FillFunctor(scalar_t value) :
        value_(value) {
    }
    DEVICE scalar_t operator()() const {
        return value_;
    }

private:
    scalar_t value_;
};

void fill_kernel(TensorIterator &iter, const any_t &value) {
    auto value_ = static_cast<double>(value);
    DISPATCH_BASIC_TYPES(iter.dtype(), "fill_kernel", [&]() {
        gpu_kernel(iter, FillFunctor<scalar_t>((scalar_t)value_));
    });
}
