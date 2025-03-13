#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"

template <typename scalar_t>
struct AddFunctor {
    DEVICE scalar_t operator()(scalar_t a, scalar_t b) const {
        return a + b;
    }
};

template <typename scalar_t>
struct SubFunctor {
    DEVICE scalar_t operator()(scalar_t a, scalar_t b) const {
        return a - b;
    }
};

template <typename scalar_t>
struct MulFunctor {
    DEVICE scalar_t operator()(scalar_t a, scalar_t b) const {
        return a * b;
    }
};

template <typename scalar_t>
struct DivFunctor {
    DEVICE scalar_t operator()(scalar_t a, scalar_t b) const {
        return a / b;
    }
};

void add_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "add_kernel", [&]() {
        gpu_kernel(iter, AddFunctor<scalar_t>());
    });
}

void sub_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "sub_kernel", [&]() {
        gpu_kernel(iter, SubFunctor<scalar_t>());
    });
}

void mul_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "mul_kernel", [&]() {
        gpu_kernel(iter, MulFunctor<scalar_t>());
    });
}

void div_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "div_kernel", [&]() {
        gpu_kernel(iter, DivFunctor<scalar_t>());
    });
}
