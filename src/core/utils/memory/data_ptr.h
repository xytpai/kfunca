#pragma once

#include <memory>

namespace utils {
namespace memory {

using DeleterFnPtr = void (*)(void *);
inline void delete_nothing(void *) {
}

class DataPtr {
private:
    void *data_;                              // Lifetime tied to ctx_
    std::unique_ptr<void, DeleterFnPtr> ctx_; // For automatic release

public:
    DataPtr() :
        data_(nullptr), ctx_(nullptr, &delete_nothing) {
    }
    DataPtr(void *data) :
        data_(data), ctx_(nullptr, &delete_nothing) {
    }
    DataPtr(void *data, void *ctx, DeleterFnPtr ctx_deleter) :
        data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &delete_nothing) {
    }
    void *operator->() const {
        return data_;
    }
    void clear() {
        ctx_ = nullptr;
        data_ = nullptr;
    }
    void *get() const {
        return data_;
    }
    void *get_context() const {
        return ctx_.get();
    }
    void *release_context() {
        return ctx_.release();
    }
    std::unique_ptr<void, DeleterFnPtr> &&move_context() {
        return std::move(ctx_);
    }
    [[nodiscard]] bool compare_exchange_deleter(
        DeleterFnPtr expected_deleter,
        DeleterFnPtr new_deleter) {
        if (get_deleter() != expected_deleter)
            return false;
        ctx_ = std::unique_ptr<void, DeleterFnPtr>(ctx_.release(), new_deleter);
        return true;
    }

    template <typename T>
    T *cast_context(DeleterFnPtr expected_deleter) const {
        if (get_deleter() != expected_deleter)
            return nullptr;
        return static_cast<T *>(get_context());
    }
    operator bool() const {
        return data_ || ctx_;
    }
    DeleterFnPtr get_deleter() const {
        return ctx_.get_deleter();
    }
};

inline bool operator==(const DataPtr &dp, std::nullptr_t) noexcept {
    return !dp;
}
inline bool operator==(std::nullptr_t, const DataPtr &dp) noexcept {
    return !dp;
}
inline bool operator!=(const DataPtr &dp, std::nullptr_t) noexcept {
    return dp;
}
inline bool operator!=(std::nullptr_t, const DataPtr &dp) noexcept {
    return dp;
}

}
} // namespace utils::memory
