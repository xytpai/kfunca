#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <ostream>
#include <cmath>
#include <bit>

#include "launcher.h"

namespace dtype {

inline float fp32_from_bits(uint32_t w) {
    return std::bit_cast<float>(w);
}

inline uint32_t fp32_to_bits(float f) {
    return std::bit_cast<uint32_t>(f);
}

inline float fp16_ieee_to_fp32_value(uint16_t h) {
    /*
     * Extend the half-precision floating-point number to 32 bits and shift to the
     * upper part of the 32-bit word:
     *      +---+-----+------------+-------------------+
     *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
     *      +---+-----+------------+-------------------+
     * Bits  31  26-30    16-25            0-15
     *
     * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
     * - zero bits.
     */
    const uint32_t w = (uint32_t)h << 16;
    /*
     * Extract the sign of the input number into the high bit of the 32-bit word:
     *
     *      +---+----------------------------------+
     *      | S |0000000 00000000 00000000 00000000|
     *      +---+----------------------------------+
     * Bits  31                 0-31
     */
    const uint32_t sign = w & UINT32_C(0x80000000);
    /*
     * Extract mantissa and biased exponent of the input number into the high bits
     * of the 32-bit word:
     *
     *      +-----+------------+---------------------+
     *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
     *      +-----+------------+---------------------+
     * Bits  27-31    17-26            0-16
     */
    const uint32_t two_w = w + w;

    /*
     * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
     * mantissa and exponent of a single-precision floating-point number:
     *
     *       S|Exponent |          Mantissa
     *      +-+---+-----+------------+----------------+
     *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
     *      +-+---+-----+------------+----------------+
     * Bits   | 23-31   |           0-22
     *
     * Next, there are some adjustments to the exponent:
     * - The exponent needs to be corrected by the difference in exponent bias
     * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
     * - Inf and NaN values in the inputs should become Inf and NaN values after
     * conversion to the single-precision number. Therefore, if the biased
     * exponent of the half-precision input was 0x1F (max possible value), the
     * biased exponent of the single-precision output must be 0xFF (max possible
     * value). We do this correction in two steps:
     *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
     * below) rather than by 0x70 suggested by the difference in the exponent bias
     * (see above).
     *   - Then we multiply the single-precision result of exponent adjustment by
     * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
     * necessary exponent adjustment by 0x70 due to difference in exponent bias.
     *     The floating-point multiplication hardware would ensure than Inf and
     * NaN would retain their value on at least partially IEEE754-compliant
     * implementations.
     *
     * Note that the above operations do not handle denormal inputs (where biased
     * exponent == 0). However, they also do not operate on denormal inputs, and
     * do not produce denormal results.
     */
    constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
    // const float exp_scale = 0x1.0p-112f;
    constexpr uint32_t scale_bits = (uint32_t)15 << 23;
    float exp_scale_val = 0;
    std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));

    const float exp_scale = exp_scale_val;
    const float normalized_value =
        fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    /*
     * Convert denormalized half-precision inputs into single-precision results
     * (always normalized). Zero inputs are also handled here.
     *
     * In a denormalized number the biased exponent is zero, and mantissa has
     * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
     *
     *                  zeros           |  mantissa
     *      +---------------------------+------------+
     *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
     *      +---------------------------+------------+
     * Bits             10-31                0-9
     *
     * Now, remember that denormalized half-precision numbers are represented as:
     *    FP16 = mantissa * 2**(-24).
     * The trick is to construct a normalized single-precision number with the
     * same mantissa and thehalf-precision input and with an exponent which would
     * scale the corresponding mantissa bits to 2**(-24). A normalized
     * single-precision floating-point number is represented as: FP32 = (1 +
     * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
     * exponent is 126, a unit change in the mantissa of the input denormalized
     * half-precision number causes a change of the constructed single-precision
     * number by 2**(-24), i.e. the same amount.
     *
     * The last step is to adjust the bias of the constructed single-precision
     * number. When the input half-precision number is zero, the constructed
     * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
     * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
     * single-precision number to get the numerical equivalent of the input
     * half-precision number.
     */
    constexpr uint32_t magic_mask = UINT32_C(126) << 23;
    constexpr float magic_bias = 0.5f;
    const float denormalized_value =
        fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    /*
     * - Choose either results of conversion of input as a normalized number, or
     * as a denormalized number, depending on the input exponent. The variable
     * two_w contains input exponent in bits 27-31, therefore if its smaller than
     * 2**27, the input is either a denormal number, or zero.
     * - Combine the result of conversion of exponent and mantissa with the sign
     * of the input number.
     */
    constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
inline uint16_t fp16_ieee_from_fp32_value(float f) {
    // const float scale_to_inf = 0x1.0p+112f;
    // const float scale_to_zero = 0x1.0p-110f;
    constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
    constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
    float scale_to_inf_val = 0, scale_to_zero_val = 0;
    std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
    std::memcpy(
        &scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
    const float scale_to_inf = scale_to_inf_val;
    const float scale_to_zero = scale_to_zero_val;

    float base = (std::abs(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<uint16_t>(
        (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

inline float f32_from_bits(uint16_t src) {
    float res = 0;
    uint32_t tmp = src;
    tmp <<= 16;
    std::memcpy(&res, &tmp, sizeof(tmp));
    return res;
}

inline uint16_t round_to_nearest_even(float src) {
    if (std::isnan(src)) {
        return UINT16_C(0x7FC0);
    } else {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        union {
            uint32_t U32; // NOLINT(facebook-hte-BadMemberName)
            float F32;    // NOLINT(facebook-hte-BadMemberName)
        };
        F32 = src;
        uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
        return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
}

class BFloat16;

struct alignas(2) Half {
    unsigned short x;
    Half() = default;
#if defined(__CUDACC__)
    DEVICE inline operator float() const;
    DEVICE inline Half(float f);
    DEVICE inline operator BFloat16() const;
#else
    inline operator float() const;
    inline Half(float f);
    inline operator BFloat16() const;
#endif
};

struct alignas(2) BFloat16 {
    unsigned short x;
    BFloat16() = default;
#if defined(__CUDACC__)
    DEVICE inline operator float() const;
    DEVICE inline BFloat16(float f);
    DEVICE inline operator Half() const;
#else
    inline operator float() const;
    inline BFloat16(float f);
    inline operator Half() const;
#endif
};

#if defined(__CUDACC__)

DEVICE inline Half::operator float() const {
    return half_to_float(x);
}

DEVICE inline Half::Half(float f) {
    x = float_to_half(f);
}

DEVICE inline Half::operator BFloat16() const {
    return BFloat16(half_to_float(x));
}

DEVICE inline BFloat16::operator float() const {
    return bfloat16_to_float(x);
}

DEVICE inline BFloat16::BFloat16(float f) {
    x = float_to_bfloat16(f);
}

DEVICE inline BFloat16::operator Half() const {
    return Half(bfloat16_to_float(x));
}

#else

inline Half::operator float() const {
    return fp16_ieee_to_fp32_value(x);
}

inline Half::Half(float f) {
    x = fp16_ieee_from_fp32_value(f);
}

inline Half::operator BFloat16() const {
    return BFloat16(fp16_ieee_to_fp32_value(x));
}

inline BFloat16::operator float() const {
    return f32_from_bits(x);
}

inline BFloat16::BFloat16(float f) {
    x = round_to_nearest_even(f);
}

inline BFloat16::operator Half() const {
    return Half(f32_from_bits(x));
}

#endif

} // namespace dtype
