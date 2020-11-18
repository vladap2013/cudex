#pragma once

namespace cudex {

// ---------- Utilities

template<typename T, size_t Extent1, size_t Extent2>
void copyHostToDevice(DeviceSpan<T, Extent1> dst, HostSpan<const T, Extent2> src)
{
    CHECK_GE(dst.size(), src.size());

    if (src.empty())
    {
        return;
    }

    cudaSafeCall(
        cudaMemcpy(dst.data(), src.data(), src.size_bytes(), cudaMemcpyHostToDevice)
    );
}

template<typename T, size_t Extent1, size_t Extent2>
void copyHostToDeviceAsync(DeviceSpan<T, Extent1> dst, HostSpan<const T, Extent2> src, cudaStream_t stream)
{
    CHECK_GE(dst.size(), src.size());

    if (src.empty())
    {
        return;
    }

    cudaSafeCall(
        cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(), cudaMemcpyHostToDevice, stream)
    );
}

template<typename T, size_t Extent1, size_t Extent2>
void copyDeviceToHost(HostSpan<T, Extent1> dst, DeviceSpan<const T, Extent2> src)
{
    CHECK_GE(dst.size(), src.size());

    if (src.empty())
    {
        return;
    }

    cudaSafeCall(
        cudaMemcpy(dst.data(), src.data(), src.size_bytes(), cudaMemcpyDeviceToHost)
    );
}

template<typename T, size_t Extent1, size_t Extent2>
void copyDeviceToHostAsync(HostSpan<T, Extent1> dst, DeviceSpan<const T, Extent2> src, cudaStream_t stream)
{
    CHECK_GE(dst.size(), src.size());

    if (src.empty())
    {
        return;
    }

    cudaSafeCall(
        cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(), cudaMemcpyHostToDevice, stream)
    );
}

template<typename T, size_t Extent1, size_t Extent2>
void copyDeviceToDevice(DeviceSpan<T, Extent1> dst, DeviceSpan<const T, Extent2> src)
{
    CHECK_GE(dst.size(), src.size());

    if (src.empty())
    {
        return;
    }

    cudaSafeCall(
        cudaMemcpy(dst.data(), src.data(), src.size_bytes(), cudaMemcpyDeviceToDevice)
    );
}

template<typename T, size_t Extent1, size_t Extent2>
void copyDeviceToDeviceAsync(DeviceSpan<T, Extent1> dst, DeviceSpan<const T, Extent2> src, cudaStream_t stream)
{
    CHECK_GE(dst.size(), src.size());

    if (src.empty())
    {
        return;
    }

    cudaSafeCall(
        cudaMemcpyAsync(dst.data(), src.data(), src.size_bytes(), cudaMemcpyDeviceToDevice, stream)
    );
}

// ---------- DeviceMemory

template<typename T>
DeviceMemory<T>::DeviceMemory()
{}

template<typename T>
DeviceMemory<T>::DeviceMemory(size_t sz)
{
    alloc(sz);
    size_ = sz;
}

template<typename T>
void DeviceMemory<T>::alloc(size_t sz)
{
    assert(data_ == nullptr);

    if (sz > 0)
    {
        cudaSafeCall(cudaMalloc(&data_, sz * sizeof(T)));
    }

    capacity_ = sz;
}

template<typename T>
void DeviceMemory<T>::free()
{
    assert(size_ <= capacity_);

    if (data_)
    {
        assert(capacity_ > 0);

        // Ignore error here, because it can report errors from previous operation.
        cudaFree(data_);

        size_ = capacity_ = 0;
        data_ = nullptr;
    }

    CHECK(data_ == nullptr);
    CHECK_EQ(size_ , 0);
    CHECK_EQ(capacity_, 0);
}

template<typename T>
DeviceMemory<T>::~DeviceMemory()
{
    free();
}

template<typename T>
DeviceSpan<T> DeviceMemory<T>::span()
{
    assert((capacity_ == 0) == (data_ == nullptr));
    assert(size_ <= capacity_);

    return {data_, size_};
}

template<typename T>
DeviceSpan<const T> DeviceMemory<T>::span() const
{
    return cspan();
}

template<typename T>
DeviceSpan<const T> DeviceMemory<T>::cspan() const
{
    assert((capacity_ == 0) == (data_ == nullptr));
    assert(size_ <= capacity_);

    return {data_, size_};
}

template<typename T>
void DeviceMemory<T>::copyFromHost(HostSpan<const T> hostSpan)
{
    copyHostToDevice(span(), hostSpan);
}

template<typename T>
void DeviceMemory<T>::copyFromHostAsync(HostSpan<const T> hostSpan)
{
    copyHostToDeviceAsync(span(), hostSpan, stream_);
}

template<typename T>
void DeviceMemory<T>::copyToHost(HostSpan<T> hostSpan) const
{
    copyDeviceToHost(hostSpan, span());
}

template<typename T>
void DeviceMemory<T>::copyToHostAsync(HostSpan<T> hostSpan) const
{
    copyDeviceToHostAsync(hostSpan, span(), stream_);
}

template<typename T>
void DeviceMemory<T>::copyFromDevice(DeviceSpan<const T> deviceSpan)
{
    copyDeviceToDevice(span(), deviceSpan);
}

template<typename T>
void DeviceMemory<T>::copyFromDeviceAsync(DeviceSpan<const T> deviceSpan)
{
    copyDeviceToDeviceAsync(span(), deviceSpan, stream_);
}

template<typename T>
size_t DeviceMemory<T>::size() const
{
    assert(size_ <= capacity_);
    return size_;
}

template<typename T>
size_t DeviceMemory<T>::capacity() const
{
    assert(size_ <= capacity_);
    return capacity_;
}

template<typename T>
DeviceSpan<T> DeviceMemory<T>::resize(size_t newSize)
{
    if (newSize > capacity()) {
        free();
        alloc(newSize);
    }

    size_ = newSize;
    return span();
}

template<typename T>
DeviceSpan<T> DeviceMemory<T>::resizeCopy(HostSpan<const T> span)
{
    resize(span.size());
    copyFromHost(span);
    return this->span();
}

template<typename T>
DeviceSpan<T> DeviceMemory<T>::resizeCopy(DeviceSpan<const T> span)
{
    resize(span.size());
    copyFromDevice(span);
    return this->span();
}

// ---------- HostDeviceMemory

template<typename T>
HostDeviceMemory<T>::HostDeviceMemory()
{}

template<typename T>
HostDeviceMemory<T>::HostDeviceMemory(size_t size)
    : device_(size)
{
    alloc();
}

template<typename T>
HostDeviceMemory<T>::HostDeviceMemory(const std::vector<T>& values)
{
    resizeSync(values);
}

template<typename T>
void HostDeviceMemory<T>::alloc()
{
    CHECK(data_ == nullptr);

    if (capacity() > 0)
    {
        cudaSafeCall(cudaMallocHost(&data_, capacity() * sizeof(T)));
    }
}

template<typename T>
void HostDeviceMemory<T>::free()
{
    if (data_)
    {
        assert(capacity() > 0);

        // Ignore error here, because it can report errors from previous operation.
        cudaFreeHost(data_);
        data_ = nullptr;
    }
}

template<typename T>
HostDeviceMemory<T>::~HostDeviceMemory()
{
    free();
}

template<typename T>
DeviceSpan<T> HostDeviceMemory<T>::copyHostToDevice(size_t size)
{
    auto dspan = device().subspan(0, size);
    auto hspan = chost().subspan(0, size);
    ::cudex::copyHostToDevice(dspan, hspan);
    return dspan;
}

template<typename T>
DeviceSpan<T> HostDeviceMemory<T>::copyHostToDeviceAsync(size_t size)
{
    auto dspan = device().subspan(0, size);
    auto hspan = chost().subspan(0, size);
    ::cudex::copyHostToDeviceAsync(dspan, hspan);
    return dspan;
}

template<typename T>
HostSpan<T> HostDeviceMemory<T>::copyDeviceToHost(size_t size)
{
    auto dspan = cdevice().subspan(0, size);
    auto hspan = host().subspan(0, size);
    ::cudex::copyDeviceToHost(hspan, dspan);
    return hspan;
}

template<typename T>
HostSpan<T> HostDeviceMemory<T>::copyDeviceToHostAsync(size_t size)
{
    auto dspan = cdevice().subspan(0, size);
    auto hspan = host().subspan(0, size);
    ::cudex::copyDeviceToHostAsync(hspan, dspan);
    return hspan;
}

template<typename T>
DeviceSpan<T> HostDeviceMemory<T>::device()
{
    return device_.span();
}

template<typename T>
DeviceSpan<const T> HostDeviceMemory<T>::device() const
{
    return cdevice();
}

template<typename T>
DeviceSpan<const T> HostDeviceMemory<T>::cdevice() const
{
    return device_.span();
}

template<typename T>
HostSpan<T> HostDeviceMemory<T>::host()
{
    assert((capacity() == 0) == (data_ == nullptr));
    return {data_, size()};
}

template<typename T>
HostSpan<const T> HostDeviceMemory<T>::host() const
{
    return chost();
}

template<typename T>
HostSpan<const T> HostDeviceMemory<T>::chost() const
{
    assert((capacity() == 0) == (data_ == nullptr));
    return {data_, size()};
}

template<typename T>
size_t HostDeviceMemory<T>::size() const
{
    return device_.size();
}

template<typename T>
size_t HostDeviceMemory<T>::capacity() const
{
    return device_.capacity();
}

template<typename T>
void HostDeviceMemory<T>::resize(size_t newSize)
{
    const size_t oldCapacity = capacity();
    device_.resize(newSize);

    assert(capacity() >= oldCapacity);

    if (oldCapacity != capacity()) {
        free();
        alloc();
    }
}

template<typename T>
void HostDeviceMemory<T>::resizeSync(const std::vector<T>& data)
{
    resize(data.size());
    std::copy(data.begin(), data.end(), data_);
    copyHostToDevice();
}

template<typename T>
T& HostDeviceMemory<T>::operator[](size_t index)
{
    return host()[index];
}

template<typename T>
T& HostDeviceMemory<T>::at(size_t index)
{
    return host()[index];
}

}

