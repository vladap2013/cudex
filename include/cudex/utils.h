#pragma once

namespace cudex {

template<typename I1, typename I2>
__host__ __device__ I1 divCeil(I1 i1, I2 i2)
{
    if constexpr (!std::is_unsigned_v<I1>)
    {
        assert(i1 >= 0);
    }

    assert(i2 > 0);

    const I1 i = static_cast<I1>(i2);
    return (i1 + i - 1) / i;
}

template<typename I1, typename I2>
__host__ __device__ auto absDiff(I1 i1, I2 i2)
{
    return (i1 > i2) ? i1 - i2 : i2 - i1;
}

template<typename I1, typename I2>
__host__ __device__ uint divDim(I1 i1, I2 i2)
{
    const auto v = divCeil(i1, i2);
    return static_cast<uint>(v);
}

template<typename T>
__host__ __device__ void swap(T& a, T& b)
{
    T c = b;
    b = a;
    a = c;
}

}


