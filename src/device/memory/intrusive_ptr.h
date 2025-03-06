#pragma once

#include <iostream>
#include <atomic>

namespace memory {

class intrusive_ptr_target;
namespace raw {
inline void incref(intrusive_ptr_target *self);
inline void decref(intrusive_ptr_target *self);
inline size_t use_count(intrusive_ptr_target *self);
} // namespace raw

class intrusive_ptr_target {
public:
    intrusive_ptr_target() noexcept :
        refcount_(0) {
    }
    ~intrusive_ptr_target() = default;

private:
    mutable std::atomic<size_t> refcount_;
    friend inline void raw::incref(intrusive_ptr_target *self);
    friend inline void raw::decref(intrusive_ptr_target *self);
    friend inline size_t raw::use_count(intrusive_ptr_target *self);
};

namespace raw {
inline size_t atomic_refcount_increment(std::atomic<size_t> &refcount) {
    return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
}
inline size_t atomic_refcount_decrement(std::atomic<size_t> &refcount) {
    return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}
inline void incref(intrusive_ptr_target *self) {
    if (self) {
        atomic_refcount_increment(self->refcount_);
    }
}
inline void decref(intrusive_ptr_target *self) {
    if (self) {
        atomic_refcount_decrement(self->refcount_);
    }
}
inline size_t use_count(intrusive_ptr_target *self) {
    if (self) {
        return self->refcount_.load(std::memory_order_acquire);
    }
    return 0;
}
} // namespace raw

template <typename T>
class intrusive_ptr {
    T *ptr_;
    inline void decref_or_delete() {
        if (ptr_) {
            raw::decref(ptr_);
            if (raw::use_count(ptr_) == 0) {
                delete ptr_;
                ptr_ = nullptr;
            }
        }
    }

public:
    size_t ref_count() const {
        return raw::use_count(ptr_);
    }
    intrusive_ptr() noexcept :
        ptr_(nullptr) {
    }
    T *get() const {
        return ptr_;
    }
    void unsafe_set_ptr(T *ptr) noexcept {
        ptr_ = ptr;
        raw::incref(ptr_);
    }
    intrusive_ptr(T *ptr) noexcept :
        ptr_(ptr) {
        raw::incref(ptr_);
    }
    intrusive_ptr(const intrusive_ptr<T> &other) noexcept :
        ptr_(other.ptr_) {
        raw::incref(ptr_);
    }
    intrusive_ptr(intrusive_ptr<T> &&other) noexcept :
        ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    intrusive_ptr<T> &operator=(const intrusive_ptr<T> &other) {
        if (ptr_ == other.ptr_) return *this;
        decref_or_delete();
        ptr_ = other.ptr_;
        raw::incref(ptr_);
        return *this;
    }
    intrusive_ptr<T> &operator=(const intrusive_ptr<T> &&other) {
        if (ptr_ == other.ptr_) return *this;
        decref_or_delete();
        ptr_ = other.ptr_;
        raw::incref(ptr_);
        return *this;
    }
    ~intrusive_ptr() {
        decref_or_delete();
    }
};

}
