// Experimental fixed routed-expert subsets for Qwen3.8-Flash-Next.
// Loaded once before any weights, caches or model state exist. An absent map is
// the unmodified model. Compact IDs are sorted original IDs, preserving ties.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/types.h>
#endif

namespace strata::core::review_subset {
struct Map {
    int64_t layers = 0, source_experts = 0, kept = 0, top_k = 0;
    std::vector<std::vector<int64_t>> ids;
    std::string fingerprint;
    bool active() const { return !ids.empty(); }
};
inline Map state;
inline bool active() { return state.active(); }
inline int64_t source_count(int64_t runtime_count) {
    return active() ? state.source_experts : runtime_count;
}
inline int64_t original(int64_t layer, int64_t compact) {
    // Callers already validate their runtime IDs before accessing a source.
    return active() ? state.ids.at(static_cast<size_t>(layer)).at(static_cast<size_t>(compact)) : compact;
}
inline bool configure(const std::string& pack, int64_t nl, int64_t ne, int64_t k, std::string& err) {
    state = {};
    const std::filesystem::path path = std::filesystem::path(pack) / "review_subset.txt";
    if (!std::filesystem::exists(path)) return true;
    auto fail = [&](const std::string& s) { err = "review subset: " + s; return false; };
    std::ifstream in(path);
    if (!in) return fail("cannot read " + path.string());
    std::string magic;
    int version = 0;
    Map next;
    if (!(in >> magic >> version >> next.layers >> next.source_experts >> next.kept >> next.top_k) ||
        magic != "STRATA_REVIEW_SUBSET" || version != 1)
        return fail("invalid header");
    if (next.layers != nl || next.source_experts != ne || next.top_k != k || next.kept < k || next.kept >= ne)
        return fail("geometry mismatch or invalid retained count; use the original pack for the full model");
    next.ids.resize(static_cast<size_t>(nl));
    for (int64_t l = 0; l < nl; ++l) {
        int64_t got_layer = -1;
        if (!(in >> got_layer) || got_layer != l) return fail("layers must be complete and ordered");
        auto& row = next.ids[static_cast<size_t>(l)];
        row.resize(static_cast<size_t>(next.kept));
        int64_t previous = -1;
        for (auto& e : row) {
            if (!(in >> e) || e <= previous || e < 0 || e >= ne)
                return fail("expert IDs must be unique, ascending, and in range");
            previous = e;
        }
    }
    std::string trailing;
    if (in >> trailing) return fail("unexpected trailing data");
    // Actual map content checksum, separate from the runner's cryptographic
    // SHA-256 over canonical JSON. Useful in the engine log without a crypto dependency.
    uint64_t hash = 14695981039346656037ull;
    auto mix = [&](uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= value & 255u; hash *= 1099511628211ull; value >>= 8;
        }
    };
    mix(next.layers); mix(next.source_experts); mix(next.kept); mix(next.top_k);
    for (const auto& row : next.ids) for (int64_t id : row) mix(id);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(hash));
    next.fingerprint = std::string("0x") + hex;
    state = std::move(next);
    return true;
}

// Read only selected expert planes. Adjacent selected IDs are coalesced into
// runs of at most 16. 'place(e)' returns the destination for compact expert e.
// This is shared by the production loader and its CPU fixture test.
template<class Place>
bool copy_role(FILE* f, uint64_t source_offset, uint64_t bytes_per_expert,
               const std::vector<int64_t>& ids, Place place, uint64_t* bytes_read = nullptr) {
    if (f == nullptr || bytes_per_expert == 0 || bytes_per_expert > std::numeric_limits<size_t>::max() / 16)
        return false;
    std::vector<uint8_t> buffer(static_cast<size_t>(bytes_per_expert * 16));
    for (size_t at = 0; at < ids.size();) {
        size_t count = 1;
        while (count < 16 && at + count < ids.size() && ids[at + count] == ids[at] + static_cast<int64_t>(count)) ++count;
        const uint64_t offset = source_offset + static_cast<uint64_t>(ids[at]) * bytes_per_expert;
        const size_t n = static_cast<size_t>(bytes_per_expert * count);
#ifdef _WIN32
        if (_fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0) return false;
#else
        if (fseeko(f, static_cast<off_t>(offset), SEEK_SET) != 0) return false;
#endif
        if (std::fread(buffer.data(), 1, n, f) != n) return false;
        if (bytes_read) *bytes_read += n;
        for (size_t j = 0; j < count; ++j)
            std::memcpy(place(static_cast<int64_t>(at + j)), buffer.data() + j * bytes_per_expert,
                        static_cast<size_t>(bytes_per_expert));
        at += count;
    }
    return true;
}
}  // namespace strata::core::review_subset
