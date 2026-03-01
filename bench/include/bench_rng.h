// bench_rng.h -- Deterministic PRNG for reproducible benchmarks
// Uses xoshiro256** (fast, deterministic, well-distributed)
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace bench {

class BenchRng {
public:
    explicit BenchRng(uint64_t seed) {
        // splitmix64 seeding
        for (auto& s : state_) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z = z ^ (z >> 31);
            s = z;
        }
    }

    uint64_t next_u64() {
        const uint64_t result = rotl(state_[1] * 5, 7) * 9;
        const uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    void fill_bytes(uint8_t* out, size_t len) {
        while (len >= 8) {
            uint64_t v = next_u64();
            std::memcpy(out, &v, 8);
            out += 8;
            len -= 8;
        }
        if (len > 0) {
            uint64_t v = next_u64();
            std::memcpy(out, &v, len);
        }
    }

    std::array<uint8_t, 32> next_32() {
        std::array<uint8_t, 32> buf{};
        fill_bytes(buf.data(), 32);
        return buf;
    }

private:
    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    uint64_t state_[4]{};
};

} // namespace bench
