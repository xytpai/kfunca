#pragma once

#include <unordered_map>
#include <limits>

#include "data_ptr.h"
#include "exception.h"

namespace utils {
namespace memory {

struct Block {
    void *ptr{nullptr};
    size_t size{0};
    int device{-1};
    bool in_use{false};
    Block *next{nullptr};
    uint32_t id;
    static uint32_t next_id;
    Block() :
        id(next_id++) {
    }
};

class DeviceAllocator {
    enum {
        POOL_SIZE = 8,
        ALIGNMENT = 1024, // 1KB
    };

    const size_t POOL_SIZE_BOUNDS[POOL_SIZE] = {
        4 * 1024,                           // 4KB
        64 * 1024,                          // 64KB
        256 * 1024,                         // 256KB
        1024 * 1024,                        // 1MB
        4 * 1024 * 1024,                    // 4MB
        64 * 1024 * 1024,                   // 64MB
        256 * 1024 * 1024,                  // 256MB
        std::numeric_limits<size_t>::max(), // inf
    };

    int _find_pool_index(size_t size) const;
    Block *_find_prev_block(size_t size);

public:
    static DeviceAllocator *
    GetInstance() {
        return m_pInstance;
    }

    void print();
    DataPtr allocate(size_t size_in_bytes, int device);
    void free(void *ptr);

private:
    DeviceAllocator() {
    }
    DeviceAllocator(const DeviceAllocator &) = delete;
    DeviceAllocator &operator=(const DeviceAllocator &) = delete;

    static DeviceAllocator *m_pInstance;

    Block pool_[POOL_SIZE];
    std::unordered_map<void *, Block *> ptr_to_block_;
};

}
} // namespace utils::memory
