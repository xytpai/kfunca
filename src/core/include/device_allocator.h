#pragma once

#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "tensor_storage.h"
#include "exception.h"

namespace utils {
namespace memory {

#define DEVICE_ALLOCATOR_MAX_COUNT 10

class DeviceAllocator {
public:
    static DeviceAllocator *
    GetInstance() {
        return m_pInstance;
    }

    void *allocate(size_t size_in_bytes, int device) {
        CHECK_FAIL(count_ < DEVICE_ALLOCATOR_MAX_COUNT);
        auto ptr = new TensorStorage(size_in_bytes, device);
        storage_[count_++].unsafe_set_ptr(ptr);
        return storage_[count_ - 1].get()->data_ptr();
    }

private:
    DeviceAllocator() {
        count_ = 0;
    }

    ~DeviceAllocator() {
    }

    DeviceAllocator(const DeviceAllocator &) = delete;
    DeviceAllocator &operator=(const DeviceAllocator &) = delete;

    static DeviceAllocator *m_pInstance;
    int count_ = 0;
    intrusive_ptr<TensorStorage> storage_[DEVICE_ALLOCATOR_MAX_COUNT];
};

}
} // namespace utils::memory
