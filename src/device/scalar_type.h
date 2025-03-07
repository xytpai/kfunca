#pragma once

#include <iostream>
#include <type_traits>

#include "exception.h"

#define FORALL_BASIC_SCALAR_TYPES(_, ...)  \
    _(uint8_t, Byte, __VA_ARGS__)  /* 0 */ \
    _(int8_t, Char, __VA_ARGS__)   /* 1 */ \
    _(int16_t, Short, __VA_ARGS__) /* 2 */ \
    _(int, Int, __VA_ARGS__)       /* 3 */ \
    _(int64_t, Long, __VA_ARGS__)  /* 4 */ \
    _(float, Float, __VA_ARGS__)   /* 5 */ \
    _(double, Double, __VA_ARGS__) /* 6 */ \
    _(bool, Bool, __VA_ARGS__)     /* 7 */

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_, n, ...) n,
    FORALL_BASIC_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
        Undefined,
    NumOptions
};

static inline const char *to_string(ScalarType t) {
#define DEFINE_CASE(_, name, ...) \
    case ScalarType::name:        \
        return #name;

    switch (t) {
        FORALL_BASIC_SCALAR_TYPES(DEFINE_CASE)
    default:
        return "UNKNOWN_SCALAR";
    }
#undef DEFINE_CASE
}

static inline size_t element_size(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name, ...) \
    case ScalarType::name:                      \
        return sizeof(ctype);

    switch (t) {
        FORALL_BASIC_SCALAR_TYPES(CASE_ELEMENTSIZE_CASE)
    default:
        CHECK_FAIL(false, "Unknown ScalarType");
    }
#undef CASE_ELEMENTSIZE_CASE
    return 0;
}

std::ostream &operator<<(std::ostream &os, const ScalarType &dtype) {
    os << to_string(dtype);
    return os;
}

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type, ...)     \
    template <>                                                        \
    struct CppTypeToScalarType<cpp_type>                               \
        : std::                                                        \
              integral_constant<ScalarType, ScalarType::scalar_type> { \
    };

FORALL_BASIC_SCALAR_TYPES(SPECIALIZE_CppTypeToScalarType)

#define DISPATCH_CASE(cpp_type, scalar_type, ...) \
    case ScalarType::scalar_type: {               \
        using scalar_t = cpp_type;                \
        return __VA_ARGS__();                     \
    }
#define DISPATCH_BASIC_TYPES(TYPE, NAME, ...)                     \
    [&] {                                                         \
        switch (TYPE) {                                           \
            FORALL_BASIC_SCALAR_TYPES(DISPATCH_CASE, __VA_ARGS__) \
        }                                                         \
    }()

#define FORALL_FLOATING_SCALAR_TYPES(_, ...) \
    _(float, Float, __VA_ARGS__)             \
    _(double, Double, __VA_ARGS__)
#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                     \
    [&] {                                                            \
        switch (TYPE) {                                              \
            FORALL_FLOATING_SCALAR_TYPES(DISPATCH_CASE, __VA_ARGS__) \
        }                                                            \
    }()

#undef SPECIALIZE_CppTypeToScalarType
