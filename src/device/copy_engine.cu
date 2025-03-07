#include <iostream>

#include "launcher.h"
#include "copy_engine.h"

void dmemcpy_h2d(void *dst, const void *src, const size_t len) {
    Launcher::GetInstance()->memcpy(dst, src, len, Launcher::COPY::H2D);
}

void dmemcpy_d2h(void *dst, const void *src, const size_t len) {
    Launcher::GetInstance()->memcpy(dst, src, len, Launcher::COPY::D2H);
}

void dmemset_zeros(void *ptr, const size_t len) {
    Launcher::GetInstance()->memset(ptr, 0, len);
}
