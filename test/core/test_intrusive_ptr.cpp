#include <iostream>
#include <cassert>

#include "data_ptr.h"
#include "intrusive_ptr.h"

using namespace utils::memory;

bool delete_flag;

void delete_ptr(void *ptr) {
    delete_flag = true;
}

class Block : public intrusive_ptr_target {
protected:
    DataPtr ptr_;

public:
    Block() :
        ptr_(nullptr) {
    }
    Block(void *ptr) {
        DataPtr data_ptr(ptr, ptr, delete_ptr);
        ptr_ = std::move(data_ptr);
    }
    ~Block() {
        ptr_.clear();
    }
    void *data_ptr() const {
        return ptr_.get();
    }
    bool defined() const {
        return static_cast<bool>(ptr_);
    }
};

int main() {
    std::cout << __FILE__ << std::endl;
    char ptr[123] = "auto";

    std::cout << "test ref count .";
    {
        delete_flag = false;
        intrusive_ptr<Block> block_ptr2;
        {
            intrusive_ptr<Block> block_ptr;
            block_ptr.unsafe_set_ptr(new Block(ptr));
            assert(block_ptr.ref_count() == 1);
            block_ptr2 = block_ptr;
            assert(block_ptr.ref_count() == 2);
            assert(block_ptr2.ref_count() == 2);
        }
        assert(block_ptr2.ref_count() == 1);
        assert(delete_flag == false);
    }
    assert(delete_flag == true);
    std::cout << " ok" << std::endl;

    std::cout << "test std::move assign .";
    { // std::move assign behaves the same as copy
        delete_flag = false;
        intrusive_ptr<Block> block_ptr2;
        {
            intrusive_ptr<Block> block_ptr;
            block_ptr.unsafe_set_ptr(new Block(ptr));
            assert(block_ptr.ref_count() == 1);
            block_ptr2 = std::move(block_ptr);
            assert(block_ptr2.get() == block_ptr.get());
            assert(block_ptr.ref_count() == 2);
            assert(block_ptr2.ref_count() == 2);
        }
        assert(block_ptr2.ref_count() == 1);
        assert(delete_flag == false);
    }
    assert(delete_flag == true);
    std::cout << " ok" << std::endl;

    std::cout << "test std::move construct .";
    { // move semantics
        delete_flag = false;
        {
            intrusive_ptr<Block> block_ptr;
            block_ptr.unsafe_set_ptr(new Block(ptr));
            assert(block_ptr.ref_count() == 1);
            intrusive_ptr<Block> block_ptr2(std::move(block_ptr));
            assert(delete_flag == false);
            assert(block_ptr.get() == nullptr);
            assert(block_ptr2.ref_count() == 1);
        }
        assert(delete_flag == true);
    }
    assert(delete_flag == true);
    std::cout << " ok" << std::endl;
}
