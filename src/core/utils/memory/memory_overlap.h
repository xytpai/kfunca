#pragma once

#include <iostream>
#include <vector>
#include <algorithm>

namespace utils {
namespace memory {

template <typename array_t>
inline bool is_non_overlapping_and_dense(array_t &shape, array_t &stride, int ndim) {
    using pair_t = std::pair<int64_t, int64_t>;
    std::vector<pair_t> v;
    for (int i = ndim - 1; i >= 0; --i)
        v.push_back(std::make_pair((int64_t)shape[i], (int64_t)stride[i]));
    std::stable_sort(v.begin(), v.end(), [](pair_t a, pair_t b) { return a.second < b.second; });
    int64_t expected_stride = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t sz = v[i].first;
        int64_t st = v[i].second;
        if (st != expected_stride) {
            return false;
        }
        expected_stride *= sz;
    }
    return true;
}

template <typename array_t>
inline std::pair<int64_t, int64_t> compute_offset_range(array_t &shape, array_t &stride, int ndim) {
    int64_t min_offset = 0;
    int64_t max_offset = 0;
    for (int i = 0; i < ndim; ++i) {
        int64_t dim = shape[i];
        int64_t s = stride[i];
        if (s >= 0) {
            max_offset += (dim - 1) * s;
        } else {
            min_offset += (dim - 1) * s;
        }
    }
    return {min_offset, max_offset};
}

template <typename array_t>
inline bool is_no_partial_overlap(
    void *self_ptr, int64_t self_element_size, array_t &self_shape, array_t &self_stride,
    void *other_ptr, int64_t other_element_size, array_t &other_shape, array_t &other_stride,
    int ndim) {
    auto [self_min, self_max] = compute_offset_range<array_t>(self_shape, self_stride, ndim);
    auto [other_min, other_max] = compute_offset_range<array_t>(other_shape, other_stride, ndim);
    char *self_ptr_min = static_cast<char *>(self_ptr) + self_min * self_element_size;
    char *self_ptr_max = static_cast<char *>(self_ptr) + self_max * self_element_size;
    char *other_ptr_min = static_cast<char *>(other_ptr) + other_min * other_element_size;
    char *other_ptr_max = static_cast<char *>(other_ptr) + other_max * other_element_size;
    auto self_ptr_min_ = reinterpret_cast<uint64_t>(self_ptr_min);
    auto self_ptr_max_ = reinterpret_cast<uint64_t>(self_ptr_max);
    auto other_ptr_min_ = reinterpret_cast<uint64_t>(other_ptr_min);
    auto other_ptr_max_ = reinterpret_cast<uint64_t>(other_ptr_max);
    return (self_ptr_max_ < other_ptr_min_ || other_ptr_max_ < self_ptr_min_);
}

}
} // namespace utils::memory
