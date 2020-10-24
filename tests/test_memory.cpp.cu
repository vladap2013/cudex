#include "gtest/gtest.h"

#include "cudex/memory.cu.h"

using namespace cudex;

namespace
{

constexpr size_t SIZE = 13;

__global__ void updateNumbers(DeviceSpan<int> span)
{
    const size_t ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < span.size())
    {
        span[ind] *= 2;
    }
}

template<typename Span>
void checkValues(const Span span, const int a, const size_t cnt)
{
    EXPECT_EQ(span.size(), cnt);

    for (size_t i = 0; i < span.size(); ++i)
    {
        EXPECT_EQ(span[i], a * (i + 1));
    }
}

}

TEST(memory_host_device, host)
{
	cudex::HostDeviceMemory<int> mem(SIZE);

    int cnt = 1;
    for (int& v : mem.host())
    {
        v = (cnt ++) * 2;
    }

    checkValues(mem.host(), 2, SIZE);
}

TEST(memory_host_device, host_device)
{
	cudex::HostDeviceMemory<int> mem(SIZE);

    int cnt = 1;
    for (int& v : mem.host())
    {
        v = (cnt ++) * 2;
    }

    checkValues(mem.host(), 2, SIZE);

    mem.copyHostToDevice();
    updateNumbers<<<2, 100>>>(mem.device());
    mem.copyDeviceToHost();

    checkValues(mem.host(), 4, SIZE);
}

TEST(memory_host_device, host_device_with_size)
{
	cudex::HostDeviceMemory<int> mem(SIZE);

    for (int& v : mem.host()) {
        v = 1;
    }

    mem.copyHostToDevice();
    updateNumbers<<<2, 100>>>(mem.device());

    static_assert(SIZE > 5);
    constexpr size_t N = SIZE - 5;

    HostSpan<int> h = mem.copyDeviceToHost(N);
    EXPECT_EQ(h.size(), N);

    size_t cnt = 0;
    for (const auto& v: h) {
        EXPECT_EQ(v, 2);
        ++cnt;
    }
    EXPECT_EQ(cnt, N);

    HostSpan<int> h2 = mem.host().subspan(N, dynamic_extent);
    EXPECT_EQ(h2.size(), SIZE - N);

    cnt = 0;
    for (const auto& v: h2) {
        EXPECT_EQ(v, 1);
        ++cnt;
    }
    EXPECT_EQ(cnt, SIZE - N);
}

TEST(memory_host_device, resizeSync)
{
    std::vector<int> vec(SIZE);

    int cnt = 1;
    for (int& v: vec) {
        v = (cnt ++) * 2;
    }

    HostDeviceMemory<int> mem;
    EXPECT_EQ(mem.size(), 0);
    EXPECT_EQ(mem.capacity(), 0);

    mem.resizeSync(vec);

    EXPECT_EQ(mem.size(), SIZE);
    EXPECT_EQ(mem[1], 4);

    checkValues(mem.host(), 2, SIZE);

    updateNumbers<<<2, 100>>>(mem.device());
    mem.copyDeviceToHost();

    checkValues(mem.host(), 4, SIZE);

    static_assert(5 < SIZE);
    mem.resize(5);
    EXPECT_EQ(mem.size(), 5);
    EXPECT_EQ(mem.capacity(), SIZE);

    checkValues(mem.host(), 4, 5);
}

