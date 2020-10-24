#include "gtest/gtest.h"

#include "cudex/memory.cu.h"
#include "cudex/launcher.cu.h"
#include "cudex/device_utils.cu.h"

using namespace cudex;

namespace
{

__global__ void setData(DeviceSpan<int> span)
{
    const size_t index = threadLinearIndex();
    if (index >= span.size())
    {
        return;
    }

    span[index] += 2;
}

}

TEST(launcher, run_1d)
{
    constexpr size_t SIZE = 1e6;
    HostDeviceMemory<int> mem(SIZE);

    EXPECT_EQ(mem.size(), SIZE);

    int cnt = 0;
    for (int& v : mem.host())
    {
        v = cnt ++;
    } 

    EXPECT_EQ(cnt, SIZE);

    mem.copyHostToDeviceAsync();

    auto launcher = Launcher().async().size1D(SIZE);
    launcher.run(setData, mem.device());

    mem.copyDeviceToHost();

    cnt = 0;
    for (const auto& v: mem.host())
    {
        EXPECT_EQ(v, cnt++ + 2);
    }

    EXPECT_EQ(cnt, SIZE);
}
