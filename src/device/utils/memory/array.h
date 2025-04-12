#pragma once

#include "launcher.h"

namespace utils {
namespace memory {

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
    T val[vec_size];
    HOST_DEVICE_INLINE T &operator[](int i) {
        return val[i];
    }
    HOST_DEVICE_INLINE T const &operator[](int i) const {
        return val[i];
    }
};

template <typename T, int vec_size>
struct array {
    T val[vec_size];
    HOST_DEVICE_INLINE T &operator[](int i) {
        return val[i];
    }
    HOST_DEVICE_INLINE T const &operator[](int i) const {
        return val[i];
    }
};

template <typename T>
struct alignas(sizeof(T) * 4) vec4 {
    union {
        T val[4];
        struct {
            T x, y, z, w;
        };
    };
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
};

template <typename T>
struct alignas(sizeof(T) * 2) vec2 {
    union {
        T val[2];
        struct {
            T x, y;
        };
    };
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
};

template <int vec_size, typename scalar_t>
DEVICE_INLINE aligned_array<scalar_t, vec_size> load_vector(const scalar_t *base_ptr, uint32_t offset) {
    using vec_t = aligned_array<scalar_t, vec_size>;
    auto *from = reinterpret_cast<const vec_t *>(base_ptr);
    return from[offset];
}

}
} // namespace utils::memory
