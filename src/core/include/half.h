#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <ostream>
#include <bit>
#include "launcher.h"

namespace dtype {

struct alignas(2) Half {
    unsigned short x;
    Half() = default;
    // Half(unsigned short bits) :
    //     x(bits) {
    // }
    HOST_DEVICE operator float() const {
        // return fp16_ieee_to_fp32_value(x);
        return 1.0;
    }
    HOST_DEVICE Half(float value) {
    }
};

} // namespace dtype
