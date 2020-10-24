#pragma once

namespace cudex {

inline __device__ size_t linearIndex(uint3 index, dim3 dimension)
{
    return index.x + (index.y * dimension.x) + (index.z * dimension.x * dimension.y);
}

inline __device__ uint3 threadMatrixIndex()
{
    return {
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z
    };
}

inline __device__ size_t threadLinearIndex()
{
    const size_t blockIndex = linearIndex(blockIdx, gridDim); 
    const size_t blockSize = blockDim.x * blockDim.y * blockDim.z;

    return blockIndex * blockSize + linearIndex(threadIdx, blockDim);
}

}
