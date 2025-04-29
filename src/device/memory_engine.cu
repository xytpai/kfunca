#include <iostream>

#include "launcher.h"
#include "memory_engine.h"

void dset_device(const int device) {
    Launcher::GetInstance()->set_device(device, false);
}

void *dmalloc(const size_t size) {
    return Launcher::GetInstance()->malloc_(size);
}

void dfree(void *ptr) {
    Launcher::GetInstance()->free(ptr);
}

void dmemcpy_h2d(void *dst, const void *src, const size_t size) {
    Launcher::GetInstance()->memcpy(dst, src, size, Launcher::COPY::H2D);
}

void dmemcpy_d2h(void *dst, const void *src, const size_t size) {
    Launcher::GetInstance()->memcpy(dst, src, size, Launcher::COPY::D2H);
}

void dmemset_zeros(void *ptr, const size_t size) {
    Launcher::GetInstance()->memset(ptr, 0, size);
}
