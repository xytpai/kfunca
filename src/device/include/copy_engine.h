#pragma once

#include <iostream>

void dmemcpy_h2d(void *dst, const void *src, const size_t len);
void dmemcpy_d2h(void *dst, const void *src, const size_t len);
void dmemset_zeros(void *ptr, const size_t len);
