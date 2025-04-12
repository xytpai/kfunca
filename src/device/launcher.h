#pragma once

#if defined(__CUDACC__)
#include "launcher_cuda.h"
#else

#ifndef HOST_DEVICE
#define HOST_DEVICE
#endif

#ifndef HOST
#define HOST
#endif

#ifndef DEVICE
#define DEVICE
#endif

#ifndef HOST_DEVICE_INLINE
#define HOST_DEVICE_INLINE
#endif

#ifndef DEVICE_INLINE
#define DEVICE_INLINE
#endif

#endif
