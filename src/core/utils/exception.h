#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <vector>

#include "xstring.h"

namespace utils {

class Error : public std::exception {
    std::string msg_;
    std::vector<std::string> context_;

    std::string backtrace_;
    std::string what_;
    const void *caller_;

    void refresh_what() {
        std::ostringstream oss;
        oss << msg_;
        if (context_.size() == 1) {
            oss << " (" << context_[0] << ")";
        } else {
            for (const auto &c : context_)
                oss << "\n  " << c;
        }
        oss << "\n"
            << backtrace_;
        what_ = oss.str();
    }

public:
    Error(
        const char *func,
        const char *file,
        const uint32_t line,
        const char *condition,
        const std::string &msg,
        const std::string &backtrace,
        const void *caller = nullptr) :
        Error(
            str("[enforce fail at ",
                file,
                ":",
                line,
                ":",
                func,
                "] ",
                condition,
                ". ",
                msg),
            backtrace,
            caller){};

    Error(
        std::string msg,
        std::string backtrace,
        const void *caller = nullptr) :
        msg_(std::move(msg)),
        backtrace_(std::move(backtrace)), caller_(caller) {
        refresh_what();
    }

    void add_context(std::string new_msg) {
        context_.push_back(std::move(new_msg));
        refresh_what();
    }
    const std::string &msg() const {
        return msg_;
    }

    const std::vector<std::string> &context() const {
        return context_;
    }

    const std::string &backtrace() const {
        return backtrace_;
    }

    const char *what() const noexcept override {
        return what_.c_str();
    }

    const void *caller() const noexcept {
        return caller_;
    }
};

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define M_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define M_UNLIKELY(expr) (expr)
#endif

inline void check_fail_(
    const char *func,
    const char *file,
    uint32_t line,
    const char *cond,
    const std::string &msg) {
    throw Error(func, file, line, cond, msg, "");
}

inline void check_fail_(
    const char *func,
    const char *file,
    uint32_t line,
    const char *cond,
    const char *msg) {
    throw Error(func, file, line, cond, std::string(msg), "");
}

template <typename... Args>
inline decltype(auto) msg_impl(const char *, const Args &...args) {
    return str(args...);
}

} // namespace utils

#define CHECK_FAIL(cond, ...)                               \
    if (M_UNLIKELY(!(cond))) {                              \
        utils::check_fail_(                                 \
            __func__,                                       \
            __FILE__,                                       \
            static_cast<uint32_t>(__LINE__),                \
            "Expected " #cond " to be true, but got false", \
            utils::msg_impl("", ##__VA_ARGS__));            \
    }
