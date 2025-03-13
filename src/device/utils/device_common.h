#pragma once

#include <iostream>

static inline int last_pow2(int n) {
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    return std::max(1, n - (n >> 1));
}

template <typename index_t>
static inline index_t div_up(index_t a, index_t b) {
    return (a + b - 1) / b;
}
