#include <iostream>
#include <cassert>

#include "device_allocator.h"

using namespace utils::memory;

int main() {
    std::cout << __FILE__ << std::endl;
    auto allocator = DeviceAllocator::GetInstance();
    {
        auto ptr = allocator->allocate(5095, 0);
    }
    auto ptr2 = allocator->allocate(4095, 0);
    auto ptr3 = allocator->allocate(4095, 0);
    allocator->print();
    auto ptr4 = allocator->allocate(5000, 0);
    allocator->print();
}
