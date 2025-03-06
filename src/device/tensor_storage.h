#pragma once

#include "data_ptr.h"
#include "intrusive_ptr.h"

namespace memory {

class TensorStorage : public intrusive_ptr_target {
protected:
    DataPtr ptr_;
    size_t size_;
    int device_;

public:
    TensorStorage() :
        ptr_(), size_(0), device_(0) {
    }
    TensorStorage(size_t size, int device);
    ~TensorStorage();
    int device() const {
        return device_;
    }
    void *data_ptr() const {
        return ptr_.get();
    }
    template <typename T>
    T *data_ptr() const {
        return reinterpret_cast<T *>(ptr_.get());
    }
    bool defined() const {
        return static_cast<bool>(ptr_);
    }
    size_t size() const {
        return size_;
    }
};

} // namespace memory
