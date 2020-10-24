#pragma once

#include <cuda.h>

#include "utils.h"

namespace cudex {

class Launcher
{
public:
    inline static constexpr uint N_BLOCK_THREADS = 256;

    Launcher() = default;
    explicit Launcher(size_t linearSize);

    template<typename F, typename... Args>
    void run(const F& f, Args&&... args) const;

    Launcher& size1D(size_t size, uint threadsPerBlock = N_BLOCK_THREADS);

    Launcher& sizeGrid(const dim3& dim);
    Launcher& sizeBlock(const dim3& dim);

    Launcher& sync();
    Launcher& async();

    Launcher& sharedMemory(size_t sz);

    Launcher& stream(cudaStream_t);

private:
    dim3 grid_;
    dim3 block_ = {N_BLOCK_THREADS, 1, 1};
    bool sync_ = false;
    size_t sharedMemory_ = 0;
    cudaStream_t stream_ = cudaStreamPerThread;
};

}

#include "launcher.inl.cu.h"

