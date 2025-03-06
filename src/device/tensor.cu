#include <iostream>
#include <vector>

#include "tensor.h"
#include "tensor_storage.h"
#include "exception.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "launcher.h"

using namespace utils::memory;

void Tensor::copy_from_cpu_ptr(void *ptr) {
    auto l = Launcher::GetInstance();
    l->memcpy(data_ptr(), ptr, storage_bytes(), Launcher::COPY::H2D);
}

void Tensor::copy_to_cpu_ptr(void *ptr) {
    auto l = Launcher::GetInstance();
    l->memcpy(ptr, data_ptr(), storage_bytes(), Launcher::COPY::D2H);
}

Tensor empty(std::vector<int64_t> shape, ScalarType dtype, int device) {
    Tensor output(shape, dtype);
    output.new_storage_(device);
    return output;
}

Tensor empty(int64_t *shape, int ndim, ScalarType dtype, int device, bool inverse) {
    Tensor output(shape, ndim, dtype, inverse);
    output.new_storage_(device);
    return output;
}

Tensor zeros(std::vector<int64_t> shape, ScalarType dtype, int device) {
    Tensor output(shape, dtype);
    output.new_storage_(device);
    Launcher::GetInstance()->memset(output.data_ptr(), 0, output.storage_bytes());
    return output;
}
