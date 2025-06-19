#pragma once

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <limits>

#include "data_ptr.h"
#include "exception.h"

namespace utils {
namespace memory {

struct Block {
    void *ptr{nullptr};
    size_t size{0};
    int device{-1};
    int stream{0};
    bool in_use{false};
    uint32_t id;
    static uint32_t next_id;
    Block() :
        id(next_id++) {
    }
    Block(size_t size, int stream) :
        size(size), stream(stream) {
    }
};

struct CompareBlock {
    bool operator()(const Block *a, const Block *b) const {
        if (a->stream != b->stream) {
            return a->stream < b->stream;
        }
        if (a->size != b->size) {
            return a->size < b->size;
        }
        return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
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

    std::set<Block *, CompareBlock> unused_blocks_[POOL_SIZE];
    std::unordered_set<Block *> active_blocks_;
    std::unordered_map<void *, Block *> ptr_to_block_;
};

}
} // namespace utils::memory
