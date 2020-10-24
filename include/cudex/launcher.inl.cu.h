#pragma once

#include "err.cu.h"

namespace cudex {

template<typename F, typename... Args>
void Launcher::run(const F& f, Args&&... args) const
{
    f<<<grid_, block_, sharedMemory_, stream_>>>(std::forward<Args>(args)...);
    checkCudaError();

    if (sync_)
    {
        syncCuda(stream_);
    }
}

inline Launcher& Launcher::sizeGrid(const dim3& dim)
{
    grid_ = dim;
    return *this;
}

inline Launcher& Launcher::sizeBlock(const dim3& dim)
{
    block_ = dim;
    return *this;
}

inline Launcher& Launcher::size1D(size_t size, uint threadsPerBlock)
{
    block_ = dim3{
        threadsPerBlock,
        1,
        1
    };

    grid_ = dim3{
        divDim(size, threadsPerBlock),
        1,
        1
    }; 

    return *this;
}

inline Launcher& Launcher::sync()
{
    sync_ = true;
    return *this;
}

inline Launcher& Launcher::async()
{
    sync_ = false;
    return *this;
}

inline Launcher& Launcher::sharedMemory(size_t size)
{
    sharedMemory_ = size;
    return *this;
}

inline Launcher& Launcher::stream(cudaStream_t stream)
{
    stream_ = stream;
    return *this;
}

inline Launcher::Launcher(size_t linearSize)
{
    size1D(linearSize);
}


}
