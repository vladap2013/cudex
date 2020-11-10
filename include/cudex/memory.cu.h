#pragma once

#include "span.cu.h"
#include "err.cu.h"

namespace cudex {

template<typename T, size_t Extent1, size_t Extent2>
void copyHostToDevice(DeviceSpan<T, Extent1> dst, HostSpan<const T, Extent2> src);

template<typename T, size_t Extent1, size_t Extent2>
void copyHostToDeviceAsync(DeviceSpan<T, Extent1> dst, HostSpan<const T, Extent2> src, cudaStream_t stream = cudaStreamPerThread);

template<typename T, size_t Extent1, size_t Extent2>
void copyDeviceToHost(HostSpan<T, Extent1> dst, DeviceSpan<const T, Extent2> src);

template<typename T, size_t Extent1, size_t Extent2>
void copyDeviceToHostAsync(HostSpan<T, Extent1> dst, DeviceSpan<const T, Extent2> src, cudaStream_t stream = cudaStreamPerThread);


template<typename T>
class DeviceMemory
{
public:
    static_assert(!std::is_const_v<T>);
    static_assert(!std::is_reference_v<T>);

    DeviceMemory();
    explicit DeviceMemory(size_t size);
    ~DeviceMemory();

    size_t size() const;
    size_t capacity() const;

    DeviceSpan<T> resize(size_t size);

    DeviceSpan<T> span();
    DeviceSpan<const T> span() const;
    DeviceSpan<const T> cspan() const;

    void copyFromHost(HostSpan<const T> span);
    void copyFromHostAsync(HostSpan<const T> span);
    void copyToHost(HostSpan<T> span) const;
    void copyToHostAsync(HostSpan<T> span) const;

    void copyFromDevice(DeviceSpan<const T> span);
    void copyFromDeviceAsync(DeviceSpan<const T> span);

    DeviceSpan<T> resizeCopy(HostSpan<const T> span);
    DeviceSpan<T> resizeCopy(DeviceSpan<const T> span);

private:
    void alloc(size_t size);
    void free();

private:
    T* data_ = nullptr;

    size_t size_ = 0;
    size_t capacity_ = 0;

    cudaStream_t stream_ = cudaStreamPerThread;
};

template<typename T>
class HostDeviceMemory
{
public:
    static_assert(!std::is_const_v<T>);
    static_assert(!std::is_reference_v<T>);

    HostDeviceMemory();
    explicit HostDeviceMemory(size_t size);
    explicit HostDeviceMemory(const std::vector<T>& values);

    ~HostDeviceMemory();

    DeviceSpan<T> copyHostToDevice(size_t count = dynamic_extent);
    DeviceSpan<T> copyHostToDeviceAsync(size_t count = dynamic_extent);

    HostSpan<T> copyDeviceToHost(size_t count = dynamic_extent);
    HostSpan<T> copyDeviceToHostAsync(size_t count = dynamic_extent);

    size_t size() const;
    size_t capacity() const;

    void resize(size_t size);

    DeviceSpan<T> device();
    DeviceSpan<const T> device() const;
    DeviceSpan<const T> cdevice() const;

    HostSpan<T> host();
    HostSpan<const T> host() const;
    HostSpan<const T> chost() const;

    T& operator[](size_t index);
    T& at(size_t index);

    void resizeSync(const std::vector<T>& data);

private:
    void alloc();
    void free();

private:
    DeviceMemory<T> device_;
    T* data_ = nullptr;
};


}

#include "memory.inl.cu.h"
