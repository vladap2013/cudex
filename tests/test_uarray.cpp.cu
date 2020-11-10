#include "gtest/gtest.h"

#include "cudex/uarray.cu.h"

using namespace cudex;

namespace
{

struct TestData
{

    static size_t cnt;

    TestData()
    {
        ++cnt;
    }

    int v = 0;

};

size_t TestData::cnt = 0;

}

TEST(uarray, test)
{
    UArray<TestData, 10> data;

    EXPECT_EQ(TestData::cnt, 0);

    TestData d1;
    EXPECT_EQ(TestData::cnt, 1);

    EXPECT_EQ(sizeof(data), 10 * sizeof(TestData));

    for (size_t i=0; i < 10; ++i) {
        data[i].v = i;
    }

    size_t cnt = 0;
    for (auto& v : data) {
        EXPECT_EQ(v.v, cnt++);
    } 
}


