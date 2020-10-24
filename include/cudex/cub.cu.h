#pragma once

#include "memory.cu.h"

#include <cub/cub.cuh>

namespace cudex {

template<typename CubF, typename... Args>
void runCub(DeviceMemory<char>& mem, CubF f, Args&&... args);

struct CubBase
{
    CubBase(cudaStream_t stream = cudaStreamPerThread)
        : tmpMem_(0)
        , cntMem_(1)
        , stream_(stream)
    {}

protected:
    DeviceMemory<char> tmpMem_;
    HostDeviceMemory<size_t> cntMem_;
    cudaStream_t stream_;
};

struct PartitionIf : public CubBase
{
    using CubBase::CubBase;

    template<typename T, typename F, typename Cnt = size_t>
    void run(DeviceSpan<T> out, DeviceSpan<const T> in, Cnt* selectedCnt, F pred);

    template<typename T, typename F>
    DeviceSpan<T> runSync(DeviceSpan<T> out, DeviceSpan<const T> in, F pred);
};


}

#include "cub.inl.cu.h"
