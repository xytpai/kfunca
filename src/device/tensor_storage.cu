#include <iostream>

#include "tensor_storage.h"
#include "exception.h"
#include "array.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "launcher.h"

namespace memory {

void delete_impl(void *ctx) {
    Launcher::GetInstance()->free(ctx);
}

TensorStorage::TensorStorage(size_t size, int device) {
    size_ = size;
    device_ = device;
    auto l = Launcher::GetInstance();
    if (l->device() != device_) l->set_device(device_);
    auto raw_ptr = (void *)l->malloc<char>(size_);
    DataPtr ptr(raw_ptr, raw_ptr, delete_impl);
    ptr_ = std::move(ptr);
}

TensorStorage::~TensorStorage() {
    auto l = Launcher::GetInstance();
    if (l->device() != device_) l->set_device(device_);
    ptr_.clear();
}

} // namespace memory
