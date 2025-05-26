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

#define CALL_VEC(VEC_SIZE)                                                                             \
    {                                                                                                  \
        using KernelT = WelfordNormPFKernel<scalar_t, acc_t, VEC_SIZE>;                                \
        auto kernel = KernelT(input_, batch_size, problem_size, eps, save_mean_, save_invstd_);        \
        kernel.init();                                                                                 \
        auto block_range = kernel.get_block_range_yx();                                                \
        auto block_size = kernel.get_block_size_yx();                                                  \
        auto allocator = DeviceAllocator::GetInstance();                                               \
        auto staging_data = allocator->allocate(kernel.staging_size() * sizeof(acc_t), self.device()); \
        auto semaphores = allocator->allocate(kernel.semaphores_size() * sizeof(int), self.device());  \
        CHECK_FAIL(kernel.set_staging_data_check((acc_t *)staging_data.get()));                        \
        kernel.set_semaphores((int *)semaphores.get());                                                \
        Launcher::GetInstance()->submit(kernel.slm_size(),                                             \
                                        {std::get<1>(block_range), std::get<0>(block_range)},          \
                                        {std::get<1>(block_size), std::get<0>(block_size)}, kernel);   \
    }

    DISPATCH_FLOATING_TYPES(dtype, "norm_stat_kernel", [&]() {
        using acc_t = acc_type<scalar_t>;
        auto input_ = (scalar_t *)self.data_ptr();
        auto save_mean_ = (acc_t *)save_mean.data_ptr();
        auto save_invstd_ = (acc_t *)save_invstd.data_ptr();
        acc_t eps = 1e-12;

        int vec_size = welford_norm_pf_kernel_vec_size<scalar_t, acc_t>(
            batch_size, input_, save_mean_, save_invstd_);

        switch (vec_size) {
        case 8:
            CALL_VEC(8)
            break;
        case 4:
            CALL_VEC(4)
            break;
        case 2:
            CALL_VEC(2)
            break;
        default:
            CALL_VEC(1)
            break;
        }
    });
#undef CALL_VEC
    return std::make_tuple(save_mean, save_invstd);
}
