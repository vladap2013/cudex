#pragma once

#include <cstdint>

namespace cudex {

template<bool isDevice, typename T, size_t extent>
constexpr Span<isDevice, T, extent>::Span()
    : detail::SpanSize<extent>(0)
    , data_(nullptr)
{}

template<bool isDevice, typename T, size_t extent>
constexpr Span<isDevice, T, extent>::Span(T* data, size_t count)
    : detail::SpanSize<extent>(count)
    , data_(data)
{
    assert(reinterpret_cast<uintptr_t>(data) % alignof(T) == 0);
}

template<bool isDevice, typename T, size_t extent>
constexpr T* Span<isDevice, T, extent>::data() const
{
    return data_;
}

template<bool isDevice, typename T, size_t extent>
constexpr T* Span<isDevice, T, extent>::begin() const
{
    return data();
}

template<bool isDevice, typename T, size_t extent>
constexpr T* Span<isDevice, T, extent>::end() const
{
    return data() + size();
}

template<bool isDevice, typename T, size_t extent>
constexpr T& Span<isDevice, T, extent>::front() const
{
    return at(0);
}

template<bool isDevice, typename T, size_t extent>
constexpr T* Span<isDevice, T, extent>::frontPtr() const
{
    return &at(0);
}

template<bool isDevice, typename T, size_t extent>
constexpr T& Span<isDevice, T, extent>::operator[](size_t ind) const
{
    return at(ind);
}

template<bool isDevice, typename T, size_t extent>
constexpr T& Span<isDevice, T, extent>::at(size_t ind) const
{
#ifndef __CUDA_ARCH__
    CHECK_LT(ind, size());
#else
    assert(ind < size());
#endif
    return data()[ind];
}

template<bool isDevice, typename T, size_t extent>
constexpr T& Span<isDevice, T, extent>::back() const
{
    return data()[size() - 1];
}

template<bool isDevice, typename T, size_t extent>
constexpr bool Span<isDevice, T, extent>::empty() const
{
    return size() == 0;
}

template<bool isDevice, typename T, size_t extent>
constexpr size_t Span<isDevice, T, extent>::size_bytes() const
{
    return size() * sizeof(T);
}

template<bool isDevice, typename T, size_t extent>
constexpr Span<isDevice, T, dynamic_extent> Span<isDevice, T, extent>::subspan(size_t offset, size_t count) const
{
    if (count == dynamic_extent) {
        count = size() - offset;
    }

#ifndef __CUDA_ARCH__
    CHECK_LE(offset + count, size());
#else
    assert(offset + count <= size());
#endif

    return {data() + offset, count};
}

template<bool isDevice, typename T, size_t extent>
constexpr Span<isDevice, T, dynamic_extent> Span<isDevice, T, extent>::first(size_t count) const
{
    return subspan(0, count);
}

template<bool isDevice, typename T, size_t extent>
constexpr Span<isDevice, T, dynamic_extent> Span<isDevice, T, extent>::head(size_t count) const
{
    return subspan(0, count);
}

template<bool isDevice, typename T, size_t extent>
constexpr Span<isDevice, T, dynamic_extent> Span<isDevice, T, extent>::tail(size_t count) const
{
    return subspan(size() - count, count);
}


template<bool isDevice, typename T, size_t extent>
template<typename Other>
constexpr Span<isDevice, Other, extent> Span<isDevice, T, extent>::cast() const
{
    static_assert(sizeof(T) == sizeof(Other));
    static_assert(std::is_const_v<T> == std::is_const_v<Other>);
    return { reinterpret_cast<Other *>(data_), size() };
}

template<typename T>
HostSpan<T> makeSpan(std::vector<T>& vec)
{
    return {vec.data(), vec.size()};
}

template<typename T>
HostSpan<const T> makeSpan(const std::vector<T>& vec)
{
    return {vec.data(), vec.size()};
}

}
