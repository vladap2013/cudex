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

    constexpr __host__ __device__ T& operator[](size_t index)
    {
        assert(index < SIZE);
        return *(begin() + index);
    }

    constexpr __host__ __device__ const T& operator[](size_t index) const
    {
        assert(index < SIZE);
        return *(begin() + index);
    }

    constexpr __host__ __device__ T* begin()
    {
        return reinterpret_cast<T*>(data_);
    }

    constexpr __host__ __device__ const T* begin() const
    {
        return reinterpret_cast<const T*>(data_);
    }

    constexpr __host__ __device__ T* end()
    {
        return begin() + SIZE;
    }

    constexpr __host__ __device__ const T* end() const
    {
        return begin() + SIZE;
    }

    constexpr __host__ __device__ void fill(const T& value)
    {
        for (T& v: *this) {
            v = value;
        }
    }

private:
    char data_[SIZE_IN_BYTES];
};

}
