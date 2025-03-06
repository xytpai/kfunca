#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace utils {

template <typename scalar_t>
struct CompareMaxdiff {
    enum {
        CPU = 0,
        CUDA
    };
    scalar_t *buf_a_, *buf_b_;
    int dev_a_, dev_b_;
    size_t n_;
    CompareMaxdiff(const scalar_t *buf_a, const int dev_a,
                   const scalar_t *buf_b, const int dev_b,
                   const size_t n) :
        buf_a_(const_cast<scalar_t *>(buf_a)),
        dev_a_(dev_a), buf_b_(const_cast<scalar_t *>(buf_b)), dev_b_(dev_b), n_(n) {
    }
    float operator()() {
        scalar_t *ptr_a, *ptr_b;
        bool is_a_new = false;
        bool is_b_new = false;
        if (dev_a_ == CPU) {
            ptr_a = buf_a_;
        } else {
            ptr_a = new scalar_t[n_];
            cudaMemcpy(ptr_a, buf_a_, n_ * sizeof(scalar_t), cudaMemcpyDeviceToHost);
            is_a_new = true;
        }
        if (dev_b_ == CPU) {
            ptr_b = buf_b_;
        } else {
            ptr_b = new scalar_t[n_];
            cudaMemcpy(ptr_b, buf_b_, n_ * sizeof(scalar_t), cudaMemcpyDeviceToHost);
            is_b_new = true;
        }
        auto maxdiff = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n_; i++) {
            auto diff = std::abs((float)ptr_a[i] - (float)ptr_b[i]);
            maxdiff = std::max(maxdiff, diff);
        }
        if (is_a_new)
            delete[] ptr_a;
        if (is_b_new)
            delete[] ptr_b;
        return maxdiff;
    }
};

template <typename T>
bool all_close(T *input, T *target, unsigned int len,
               double atol = 1e-5, double rtol = 1e-5, int max_errors_print = 10) {
    unsigned int errors = 0;
    for (unsigned int i = 0; i < len; i++) {
        if (std::isnan(input[i]) || (std::abs(input[i] - target[i]) > atol + rtol * std::abs(target[i]))) {
            if (errors < max_errors_print)
                std::cout << "Accuracy error: index[" << i << "], input: " << input[i] << ", target: " << target[i] << std::endl;
            errors++;
        }
    }
    bool is_error = errors != 0;
    if (is_error)
        std::cout << "Total " << errors << " errors\n";
    return !is_error;
}

} // namespace utils
