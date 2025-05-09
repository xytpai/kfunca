#include "tensor_iterator.h"
#include "welford_norm.h"
#include "scalar_type.h"
#include "accumulate_type.h"

std::tuple<Tensor, Tensor> norm_stat_kernel(const Tensor &self, const int dim) {
    CHECK_FAIL(self.defined());
    CHECK_FAIL(dim == 0 && self.dim() == 2);
    // CHECK_FAIL(self.is_contiguous());
    auto problem_size = self.shape(dim);
    auto batch_size = self.numel() / problem_size;
    auto dtype = self.dtype();
    auto acc_dtype = accumulate_type(dtype);
    auto save_mean = empty_like_reduced(self, dim, acc_dtype);
    auto save_invstd = empty_like_reduced(self, dim, acc_dtype);

    DISPATCH_FLOATING_TYPES(dtype, "norm_stat_kernel", [&]() {
        using acc_t = acc_type<scalar_t>;
        using KernelT = WelfordNormPFKernel<scalar_t, acc_t, 4>;
        auto input_ = (scalar_t *)self.data_ptr();
        auto save_mean_ = (acc_t *)save_mean.data_ptr();
        auto save_invstd_ = (acc_t *)save_invstd.data_ptr();
        auto valid = KernelT::valid(batch_size, problem_size, input_,
                                    save_mean_,
                                    save_invstd_);
        CHECK_FAIL(valid == true);
        auto kernel = KernelT(input_,
                              save_mean_,
                              save_invstd_,
                              batch_size,
                              problem_size);
        kernel.init();
        auto block_range = kernel.get_block_range_yx();
        auto block_size = kernel.get_block_size_yx();
        Launcher::GetInstance()->submit(kernel.slm_size(),
                                        {std::get<1>(block_range), std::get<0>(block_range)},
                                        {std::get<1>(block_size), std::get<0>(block_size)}, kernel);
    });
    return std::make_tuple(save_mean, save_invstd);
}
