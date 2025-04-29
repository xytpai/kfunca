#include <iostream>

#include "device_allocator.h"
#include "memory_engine.h"

using namespace utils::memory;

static void delete_impl(void *ptr);

int DeviceAllocator::_find_pool_index(size_t size) const {
    int idx;
    for (idx = 0; idx < POOL_SIZE; idx++) {
        if (POOL_SIZE_BOUNDS[idx] > size) {
            break;
        }
    }
    return idx;
}

Block *DeviceAllocator::_find_prev_block(size_t size) {
    Block *current = &pool_[_find_pool_index(size)];
    while (current->next != nullptr && size > current->next->size) {
        current = current->next;
    }
    while (current->next != nullptr && size == current->next->size && current->next->in_use) {
        current = current->next;
    }
    return current;
}

void DeviceAllocator::print() {
    size_t l_size = 0;
    size_t h_size = 0;
    for (int p = 0; p < POOL_SIZE; p++) {
        l_size = h_size;
        h_size = POOL_SIZE_BOUNDS[p];
        std::cout << "[" << l_size << ", " << h_size << "):";
        Block *current = &pool_[p];
        while (current != nullptr) {
            std::cout << current->id << ":" << current->size << ":" << current->in_use << "->";
            current = current->next;
        }
        std::cout << std::endl;
    }
}

DataPtr DeviceAllocator::allocate(const size_t size, int device) {
    dset_device(device);
    size_t aligned_size = (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    Block *prev_block_ptr = _find_prev_block(aligned_size);
    if (prev_block_ptr->next && prev_block_ptr->next->size == aligned_size
        && !prev_block_ptr->next->in_use) {
        prev_block_ptr->next->in_use = true;
        return {prev_block_ptr->next->ptr, prev_block_ptr->next->ptr, delete_impl};
    } else {
        auto raw_ptr = dmalloc(aligned_size);
        auto new_block_ptr = new Block();
        new_block_ptr->ptr = raw_ptr;
        new_block_ptr->size = aligned_size;
        new_block_ptr->device = device;
        new_block_ptr->in_use = true;
        new_block_ptr->next = prev_block_ptr->next;
        prev_block_ptr->next = new_block_ptr;
        ptr_to_block_[raw_ptr] = new_block_ptr;
        return {new_block_ptr->ptr, new_block_ptr->ptr, delete_impl};
    }
    return {nullptr, nullptr, delete_nothing};
}

void DeviceAllocator::free(void *ptr) {
    Block *block_ptr = ptr_to_block_[ptr];
    block_ptr->in_use = false;
}

DeviceAllocator *DeviceAllocator::m_pInstance = new DeviceAllocator();

static void delete_impl(void *ptr) {
    DeviceAllocator::GetInstance()->free(ptr);
}
