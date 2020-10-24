#pragma once

#include <glog/logging.h>

#include <cassert>
#include <vector>

namespace cudex {

inline constexpr size_t dynamic_extent = static_cast<size_t>(-1);

namespace detail
{

template<size_t extent>
class SpanSize
{
    static_assert(extent != 0);

    constexpr __host__ __device__ SpanSize(size_t count)
    {
        (void) count;
        assert(count == extent);
    }

    constexpr __host__ __device__ size_t size() const
    {
        return extent;
    }
};

template<>
class SpanSize<dynamic_extent>
{
public:
    constexpr __host__ __device__ SpanSize(size_t count) : size_(count)
    {
        assert(count != dynamic_extent);
    }

    constexpr __host__ __device__ size_t size() const
    {
        return size_;
    }

protected:
    size_t size_;
};


} // namespace detail


template<bool isDevice, typename T, size_t extent = dynamic_extent>
class Span final : public detail::SpanSize<extent>
{
public:
    using NonConstType = std::remove_const_t<T>;
    using ConstType = const NonConstType;

    using detail::SpanSize<extent>::size;

    template<typename S>
    constexpr static bool isSameType = std::is_same<S, NonConstType>::value || std::is_same<S, ConstType>::value;

    constexpr __host__ __device__ Span();
    constexpr __host__ __device__ Span(T* data, size_t count);

    template<typename S, size_t extent2, typename = std::enable_if_t<isSameType<S>>>
    constexpr __host__ __device__ Span(Span<isDevice, S, extent2> other) : Span(other.data(), other.size())
    {
        static_assert(extent == dynamic_extent || extent2 == dynamic_extent || extent == extent2);
    }

    constexpr __host__ __device__ Span<isDevice, ConstType, extent> cspan() const
    {
        return {data_, size()};
    }

    constexpr __host__ __device__ bool empty() const;
    constexpr __host__ __device__ size_t size_bytes() const;

    constexpr __host__ __device__ T* data() const;

    constexpr __host__ __device__ T* begin() const;
    constexpr __host__ __device__ T* end() const;

    constexpr __host__ __device__ T& front() const;
    constexpr __host__ __device__ T& back() const;

    constexpr __host__ __device__ T* frontPtr() const;

    constexpr __host__ __device__ T& operator[](size_t index) const;
    constexpr __host__ __device__ T& at(size_t index) const;

    constexpr __host__ __device__ Span<isDevice, T, dynamic_extent> subspan(size_t offset, size_t count) const;
    constexpr __host__ __device__ Span<isDevice, T, dynamic_extent> first(size_t cnt) const;
    constexpr __host__ __device__ Span<isDevice, T, dynamic_extent> head(size_t cnt) const;
    constexpr __host__ __device__ Span<isDevice, T, dynamic_extent> last(size_t cnt) const;
    constexpr __host__ __device__ Span<isDevice, T, dynamic_extent> tail(size_t cnt) const;

    template<typename Other>
    constexpr Span<isDevice, Other, extent> cast() const;

private:
    T* data_;
};


template<typename T, size_t extent = dynamic_extent>
using DeviceSpan = Span<true, T, extent>;

template<typename T, size_t extent = dynamic_extent>
using HostSpan = Span<false, T, extent>;


// Helper functions

template<typename T>
HostSpan<T> makeSpan(std::vector<T>& vector);

template<typename T>
HostSpan<const T> makeSpan(const std::vector<T>& vector);

}

#include "span.inl.cu.h"
