#pragma once

#include <iostream>
#include <string>
#include <sstream>

namespace utils {

inline std::ostream &_str(std::ostream &ss) {
    return ss;
}

template <typename T>
inline std::ostream &_str(std::ostream &ss, const T &t) {
    ss << t;
    return ss;
}

template <typename T, typename... Args>
inline std::ostream &_str(std::ostream &ss, const T &t, const Args &...args) {
    return _str(_str(ss, t), args...);
}

template <typename... Args>
struct _str_wrapper final {
    static std::string call(const Args &...args) {
        std::ostringstream ss;
        _str(ss, args...);
        return ss.str();
    }
};

template <>
struct _str_wrapper<std::string> final {
    static const std::string &call(const std::string &str) {
        return str;
    }
};

template <>
struct _str_wrapper<const char *> final {
    static const char *call(const char *str) {
        return str;
    }
};

template <typename... Args>
inline decltype(auto) str(const Args &...args) {
    return _str_wrapper<Args...>::call(args...);
}

}
