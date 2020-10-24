#pragma once

#include <cuda.h>
#include <stdexcept>

#define cudaSafeCall(call) \
    do { \
        const cudaError_t err = call; \
        if (!__builtin_expect(err == cudaSuccess, 1)) \
        {\
            throw CudaAPIException(err, __FILE__, __LINE__); \
        }\
    } while(0)


namespace cudex {

class CudaAPIException: public std::runtime_error
{
public:
    CudaAPIException(cudaError_t err, const char* file, int line)
        : std::runtime_error(makeWhat(err, file, line))
        , err_(err)
    {}

private:
    std::string makeWhat(cudaError_t err, const char* file, int line) const
    {
        std::stringstream buf;
        buf.precision(20);

        buf << "cuda ERROR: "
            << cudaGetErrorString(err) << "(" << err << ") at: "
            << file << "(" << line << ")";

        return buf.str();
    }

    const cudaError_t err_;
};

inline void syncCuda(cudaStream_t stream = cudaStreamPerThread)
{
    cudaSafeCall(cudaStreamSynchronize(stream));
}

inline void checkCudaError()
{
    cudaSafeCall(cudaPeekAtLastError());
}


}
