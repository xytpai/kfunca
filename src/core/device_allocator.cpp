#include <iostream>
#include <algorithm>

#include "device_allocator.h"
#include "memory_engine.h"

using namespace utils::memory;

static void delete_impl(void *ptr);

uint32_t Block::next_id = 0;

int DeviceAllocator::_find_pool_index(size_t size) const {
    return std::lower_bound(POOL_SIZE_BOUNDS, POOL_SIZE_BOUNDS + POOL_SIZE, size) - POOL_SIZE_BOUNDS;
}

void DeviceAllocator::print() {
    std::cout << "Unused Blocks:\n";
    size_t l_size = 0;
    size_t h_size = 0;
    for (int p = 0; p < POOL_SIZE; p++) {
        l_size = h_size;
        h_size = POOL_SIZE_BOUNDS[p];
        std::cout << "[" << l_size << ", " << h_size << "):";
        for (auto item : unused_blocks_[p]) {
            std::cout << item->id << ":" << item->size << ":" << item->in_use << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "Active Blocks:\n";
    for (auto item : active_blocks_) {
        std::cout << item->id << ":" << item->size << ":" << item->in_use << ", ";
    }
    std::cout << std::endl;
}

DataPtr DeviceAllocator::allocate(const size_t size, int device, int stream) {
    dset_device(device);
    size_t aligned_size = (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    auto &pool = unused_blocks_[_find_pool_index(size)];
    auto search_key = Block(size, stream);
    auto it = pool.lower_bound(&search_key);
    if (it == pool.end()) {
        auto raw_ptr = dmalloc(aligned_size);
        auto new_block_ptr = new Block();
        new_block_ptr->ptr = raw_ptr;
        new_block_ptr->size = aligned_size;
        new_block_ptr->device = device;
        new_block_ptr->stream = stream;
        new_block_ptr->in_use = true;
        ptr_to_block_[raw_ptr] = new_block_ptr;
        bool inserted = active_blocks_.insert(new_block_ptr).second;
        CHECK_FAIL(inserted);
        return {new_block_ptr->ptr, new_block_ptr->ptr, delete_impl};
    } else {
        (*it)->in_use = true;
        pool.erase(it);
        bool inserted = active_blocks_.insert((*it)).second;
        CHECK_FAIL(inserted);
        return {(*it)->ptr, (*it)->ptr, delete_impl};
    }
    return {nullptr, nullptr, delete_nothing};
}

void DeviceAllocator::free(void *ptr) {
    Block *block_ptr = ptr_to_block_[ptr];
    block_ptr->in_use = false;
    active_blocks_.erase(block_ptr);
    auto size = block_ptr->size;
    bool inserted = unused_blocks_[_find_pool_index(size)].insert(block_ptr).second;
    CHECK_FAIL(inserted);
}

DeviceAllocator *DeviceAllocator::m_pInstance = new DeviceAllocator();

static void delete_impl(void *ptr) {
    DeviceAllocator::GetInstance()->free(ptr);
}
