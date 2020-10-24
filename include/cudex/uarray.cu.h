#pragma once

namespace cudex {

/*
 * Fixed size array which does not initialize members
 */
template<typename T, size_t SIZE>
class alignas(T) UArray
{
public:
    static constexpr size_t SIZE_IN_BYTES = SIZE * sizeof(T);

    __host__ __device__ T& operator[](size_t index)
    {
        return *(begin() + index);
    }

    __host__ __device__ const T& operator[](size_t index) const
    {
        return *(begin() + index);
    }

    __host__ __device__ T* begin()
    {
        return reinterpret_cast<T*>(data_);
    }

    __host__ __device__ const T* begin() const
    {
        return reinterpret_cast<T*>(data_);
    }

    __host__ __device__ T* end()
    {
        return begin() + SIZE;
    }

    __host__ __device__ const T* end() const
    {
        return begin() + SIZE;
    }

private:
    char data_[SIZE_IN_BYTES];
};

}
