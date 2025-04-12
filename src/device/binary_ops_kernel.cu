#include "tensor_iterator.h"
#include "tensor_loops.h"
#include "scalar_type.h"
#include "accumulate_type.h"

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
        using acc_t = acc_type<scalar_t>;
        gpu_kernel(iter, AddFunctor<acc_t>());
    });
}

void sub_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "sub_kernel", [&]() {
        using acc_t = acc_type<scalar_t>;
        gpu_kernel(iter, SubFunctor<acc_t>());
    });
}

void mul_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "mul_kernel", [&]() {
        using acc_t = acc_type<scalar_t>;
        gpu_kernel(iter, MulFunctor<acc_t>());
    });
}

void div_kernel(TensorIterator &iter) {
    DISPATCH_BASIC_TYPES(iter.common_dtype(), "div_kernel", [&]() {
        using acc_t = acc_type<scalar_t>;
        gpu_kernel(iter, DivFunctor<acc_t>());
    });
}
