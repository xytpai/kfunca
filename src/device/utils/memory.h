#pragma once

namespace utils {

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

}
