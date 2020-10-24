#pragma once

#include <cuda.h>

namespace cudex {

inline __device__ uint3 threadMatrixIndex();

inline __device__ size_t threadLinearIndex();

}


#include "device_utils.inl.cu.h"

