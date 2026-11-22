#include "strata/core/review_subset.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
using namespace strata::core::review_subset;
int main(int argc, char** argv) {
    assert(argc == 2);
    std::string folder = argv[1], error;
    assert(configure(folder, 2, 8, 2, error));
    assert(!active() && original(1, 6) == 6 && source_count(8) == 8);
    {
        std::ofstream f(folder + "/review_subset.txt");
        f << "STRATA_REVIEW_SUBSET 1\n2 8 4 2\n0 0 2 3 7\n1 1 4 5 6\n";
    }
    assert(configure(folder, 2, 8, 2, error));
    assert(active() && source_count(4) == 8 && original(0, 3) == 7 && original(1, 0) == 1);
    assert(state.fingerprint == "0x3dbbe055fa9955c9");
    constexpr uint64_t stride = 13, prefix = 19, dst_stride = 31;
    std::vector<uint8_t> src(prefix + 8 * stride);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i);
    FILE* f = std::tmpfile();
    assert(f && std::fwrite(src.data(), 1, src.size(), f) == src.size());
    std::vector<uint8_t> dst(4 * dst_stride, 0xaa);
    uint64_t read_bytes = 0;
    assert(copy_role(f, prefix, stride, state.ids[0], [&](int64_t e) { return dst.data() + e * dst_stride; }, &read_bytes));
    assert(read_bytes == 4 * stride); // no discarded expert plane was requested
    for (int e = 0; e < 4; ++e) {
        for (uint64_t j = 0; j < stride; ++j)
            assert(dst[e * dst_stride + j] == src[prefix + state.ids[0][e] * stride + j]);
        for (uint64_t j = stride; j < dst_stride; ++j) assert(dst[e * dst_stride + j] == 0xaa);
    }
    assert(!copy_role(f, src.size(), stride, state.ids[0], [&](int64_t e) { return dst.data() + e * dst_stride; }));
    std::fclose(f);
    const std::string fixture = folder + "/roles.bin";
    { std::ofstream out(fixture, std::ios::binary); out.write(reinterpret_cast<const char*>(src.data()), src.size()); }
    // Several request/loading threads share one immutable map and use independent
    // I/O handles/scratch. This validates the helper's thread safety, not GPU slots.
    std::atomic<int> completed{0};
    std::vector<std::thread> readers;
    for (int worker = 0; worker < 4; ++worker) readers.emplace_back([&, worker] {
        FILE* handle = std::fopen(fixture.c_str(), "rb");
        assert(handle);
        std::vector<uint8_t> result(4 * stride);
        for (int repeat = 0; repeat < 20; ++repeat) {
            const int layer = worker % 2;
            assert(copy_role(handle, prefix, stride, state.ids[layer], [&](int64_t e) { return result.data() + e * stride; }));
            for (int e = 0; e < 4; ++e)
                for (uint64_t j = 0; j < stride; ++j)
                    assert(result[e * stride + j] == src[prefix + original(layer,e) * stride + j]);
        }
        std::fclose(handle);
        ++completed;
    });
    for (auto& reader : readers) reader.join();
    assert(completed == 4);
    {
        std::ofstream bad(folder + "/review_subset.txt");
        bad << "STRATA_REVIEW_SUBSET 1\n2 8 4 2\n0 0 2 2 7\n1 1 4 5 6\n";
    }
    assert(!configure(folder, 2, 8, 2, error));
    assert(!active());
    std::cout << "PASS: identity mode, strict mapping, sparse plane reads, padding, short read, concurrent reads and duplicate rejection\n";
}
