#pragma once

namespace cudex {

template<typename T, size_t size>
class Stack
{
public:
    __device__ void push(const T& value) {
        assert(sp_ < size);
        data_[sp_++] = value;
    }

    __device__ const T& top() const {
        assert(sp_ > 0);
        return data_[sp_ - 1];
    };

    __device__ T& top() {
        assert(sp_ > 0);
        return data_[sp_ - 1];
    }

    __device__ void pop() {
        assert(sp_ > 0);
        --sp_;
    }

    __device__ bool empty() const {
        return sp_ == 0;
    }

private:
    T data_[size];
    size_t sp_ = 0;
};

}
