#pragma once

#include "scalar_type.h"

template <typename T>
struct AccumulateType {};

#define ACC_TYPE(t, acc_t)     \
    template <>                \
    struct AccumulateType<t> { \
        using type = acc_t;    \
    };

template <typename T>
using acc_type = typename AccumulateType<T>::type;

ACC_TYPE(dtype::Half, float)
ACC_TYPE(float, float)
ACC_TYPE(double, double)
ACC_TYPE(int8_t, int64_t)
ACC_TYPE(uint8_t, int64_t)
ACC_TYPE(char, int64_t)
ACC_TYPE(int16_t, int64_t)
ACC_TYPE(int32_t, int64_t)
ACC_TYPE(int64_t, int64_t)
ACC_TYPE(bool, bool)
