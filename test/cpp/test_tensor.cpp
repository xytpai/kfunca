#include <iostream>
#include <vector>
#include <cassert>

#include "tensor.h"

int main() {
    std::cout << __FILE__ << std::endl;
    std::vector<int64_t> shape = {3, 5};
    int tensor_data[15] = {
        3, 2, -1, 2, 3, // 0
        5, 4, 3, 4, 5,  // 1
        2, 4, 6, 8, 6,  // 2
    };
    auto t = empty(shape, ScalarType::Int, 0);
    t.copy_from_cpu_ptr(tensor_data);
    t = t + t;
    int out[15] = {0};
    t.copy_to_cpu_ptr(out);
    for (int i = 0; i < 15; i++) {
        assert(tensor_data[i] == out[i] / 2);
    }
}
