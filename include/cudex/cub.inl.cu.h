#pragma once

#include <glog/logging.h>

namespace cudex {

template<typename CubF, typename... Args>
void runCub(DeviceMemory<char>& mem, CubF f, Args&&... args)
{
    size_t size;
    void *tmpStorage = 0;
    while(true)
    {
        cudaSafeCall(
            f(tmpStorage, size, std::forward<Args>(args)...)
        );

        if (tmpStorage)
        {
            break;
        }
        else
        {
            mem.resize(size);
            tmpStorage = mem.span().begin();
        }
    }
}

template<typename T, typename F, typename Cnt>
void PartitionIf::run(DeviceSpan<T> out, DeviceSpan<const T> in, Cnt* selectedCnt, F pred)
{
    CHECK_LE(in.size(), out.size());
    CHECK_GT(in.size(), 0);

    auto fcub = cub::DevicePartition::If<const T*, T*, Cnt*, F>;

    runCub(tmpMem_, fcub, in.begin(), out.begin(), selectedCnt, in.size(), pred, stream_, false);
}

template<typename T, typename F>
DeviceSpan<T> PartitionIf::runSync(DeviceSpan<T> out, DeviceSpan<const T> in, F pred)
{
    cntMem_.resize(1);

    run(out, in, cntMem_.device().frontPtr(), std::move(pred));
    cntMem_.copyDeviceToHost();

    CHECK_LE(cntMem_.host()[0], in.size());

    return DeviceSpan<T>(out.data(), cntMem_.host()[0]);
}

template<typename T, typename F>
void Reduce::run(DeviceSpan<const T> data, F oper, T init, T* out)
{
    auto fcub = cub::DeviceReduce::Reduce<const T*, T*, F, T>;
    runCub(tmpMem_, fcub, data.data(), out, data.size(), oper, init, stream_, false);
}

template<typename T, typename D, typename FTransform, typename F>
void Reduce::runTransformed(DeviceSpan<const D> data, FTransform ftransform, F oper, T init, T* out)
{
    using It = cub::TransformInputIterator<T, FTransform, const D*>;
    It it(data.data(), ftransform);

    auto fcub = cub::DeviceReduce::Reduce<It, T*, F, T>;
    runCub(tmpMem_, fcub, it, out, data.size(), oper, init, stream_, false);
}

}
