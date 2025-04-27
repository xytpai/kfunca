#pragma once

#include <iostream>

void *dmalloc(const size_t size);
void dfree(void *ptr);
void dmemcpy_h2d(void *dst, const void *src, const size_t size);
void dmemcpy_d2h(void *dst, const void *src, const size_t size);
void dmemset_zeros(void *ptr, const size_t size);
