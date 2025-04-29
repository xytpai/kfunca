#include <iostream>
#include <cassert>

#include "data_ptr.h"

bool delete_flag;

void delete_ptr(void *ptr) {
    delete_flag = true;
}

int main() {
    std::cout << __FILE__ << std::endl;
    char ptr[123];

    std::cout << "test delete .";
    delete_flag = false;
    {
        auto data_ptr = utils::memory::DataPtr(ptr, ptr, delete_ptr);
        assert(delete_flag == false);
    }
    assert(delete_flag == true);
    std::cout << ".";
    delete_flag = false;
    {
        auto data_ptr = utils::memory::DataPtr(ptr, ptr, delete_ptr);
        assert(delete_flag == false);
        data_ptr.clear();
        assert(delete_flag == true);
    }
    std::cout << " ok" << std::endl;

    std::cout << "test get .";
    {
        ptr[0] = 'a';
        ptr[1] = 'u';
        ptr[2] = 't';
        ptr[3] = 'o';
        ptr[4] = '\0';
        auto data_ptr = utils::memory::DataPtr(ptr, ptr, delete_ptr);
        auto char_ptr = (char *)data_ptr.get();
        assert(char_ptr[0] == 'a');
        assert(char_ptr[1] == 'u');
        assert(char_ptr[2] == 't');
        assert(char_ptr[3] == 'o');
        assert((bool)data_ptr == true);
    }
    std::cout << " ok" << std::endl;

    std::cout << "test release context .";
    delete_flag = false;
    {
        auto data_ptr = utils::memory::DataPtr(ptr, ptr, delete_ptr);
        data_ptr.release_context();
        assert(delete_flag == false);
    }
    assert(delete_flag == false);
    std::cout << " ok" << std::endl;

    std::cout << "test std::move .";
    delete_flag = false;
    {
        auto data_ptr = utils::memory::DataPtr(ptr, ptr, delete_ptr);
        assert(data_ptr.get_context() != nullptr);
        auto data_ptr2 = std::move(data_ptr);
        assert(delete_flag == false);
        assert(data_ptr.get_context() == nullptr);
        assert(data_ptr2 != nullptr);
        assert(data_ptr.get() == data_ptr2.get());
        auto p = (char *)data_ptr2.get();
        assert(p[0] == 'a');
        assert(p[1] == 'u');
        assert(p[2] == 't');
        assert(p[3] == 'o');
    }
    assert(delete_flag == true);
    std::cout << " ok" << std::endl;
}
