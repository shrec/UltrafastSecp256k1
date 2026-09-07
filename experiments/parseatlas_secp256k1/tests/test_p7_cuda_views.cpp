// P7 finite, independent host oracle. No timings, CT certification, or claim
// that host canaries constitute a device-memory/aliasing sanitizer.
#include "../probes/p7_cuda_views.hpp"
#include "secp256k1/field.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using boost::multiprecision::cpp_int;
using FE = secp256k1::fast::FieldElement;
using Limbs = std::array<std::uint64_t, 4>;
using Wide = std::array<std::uint64_t, 8>;
using Bytes = std::array<std::uint8_t, 32>;
const cpp_int kB = cpp_int(1) << 256;
const cpp_int kP = kB - (cpp_int(1) << 32) - 977;
constexpr std::uint64_t kSeed = UINT64_C(0x5041375f56494557);
constexpr std::size_t kRandomPairs = 2048, kChainLanes = 17;
constexpr unsigned kChainLength = 256;
static_assert(sizeof(P7Field) == 32 && std::is_trivially_copyable_v<P7Field>);

struct Counts {
    std::uint64_t assertions = 0, cpu_checks = 0, field_outputs = 0, raw_outputs = 0;
    std::uint64_t eval_calls = 0, raw_calls = 0, error_rejections = 0;
    std::uint64_t preservation_checks = 0, canary_checks = 0, negative_controls = 0;
    std::uint64_t checksum = UINT64_C(14695981039346656037);
    std::array<std::uint64_t, 4> field_by_route{}, raw_by_route{};
};
void require(bool value, Counts& c, const std::string& context) {
    ++c.assertions;
    if (!value) throw std::runtime_error(context);
}
cpp_int mod(cpp_int x) { x %= kP; if (x < 0) x += kP; return x; }
template<std::size_t N> std::array<std::uint64_t, N> words(cpp_int x) {
    if (x < 0 || x >= (cpp_int(1) << (64 * N))) throw std::runtime_error("oracle word range");
    std::array<std::uint64_t, N> out{};
    for (auto& v : out) { v = static_cast<std::uint64_t>(x & ((cpp_int(1) << 64) - 1)); x >>= 64; }
    return out;
}
template<std::size_t N> cpp_int integer(const std::array<std::uint64_t, N>& x) {
    cpp_int out = 0;
    for (std::size_t i = N; i; --i) { out <<= 64; out += x[i - 1]; }
    return out;
}
template<std::size_t N> std::array<std::uint8_t, N> bytes(cpp_int x) {
    if (x < 0 || x >= (cpp_int(1) << (8 * N))) throw std::runtime_error("oracle byte range");
    std::array<std::uint8_t, N> out{};
    for (std::size_t i = N; i; --i) { out[i - 1] = static_cast<std::uint8_t>(x & 255); x >>= 8; }
    return out;
}
template<std::size_t N> std::array<std::uint8_t, N * 8> serialize(const std::array<std::uint64_t, N>& x) {
    std::array<std::uint8_t, N * 8> out{};
    for (std::size_t i = 0; i < N; ++i)
        for (unsigned j = 0; j < 8; ++j) out[N * 8 - 1 - (i * 8 + j)] = static_cast<std::uint8_t>(x[i] >> (j * 8));
    return out;
}
template<std::size_t N> std::string hex(const std::array<std::uint8_t, N>& x) {
    std::ostringstream out; out << std::hex << std::setfill('0');
    for (auto v : x) out << std::setw(2) << unsigned(v);
    return out.str();
}
Limbs limbs(const P7Field& x) { return {x.limbs[0], x.limbs[1], x.limbs[2], x.limbs[3]}; }
P7Field field(const Limbs& x) { return {{x[0], x[1], x[2], x[3]}}; }
P7Field field(const cpp_int& x) { return field(words<4>(x)); }
template<std::size_t N> void mix(const std::array<std::uint8_t, N>& x, Counts& c) {
    for (auto v : x) c.checksum = (c.checksum ^ v) * UINT64_C(1099511628211);
}
bool canonical(const Limbs& x) {
    static const Limbs p = words<4>(kP);
    for (std::size_t i = 4; i; --i) if (x[i - 1] != p[i - 1]) return x[i - 1] < p[i - 1];
    return false;
}
struct Expected { Limbs raw; Bytes encoded; Limbs cpu_raw; Bytes cpu_encoded; };
bool field_matches(const Limbs& raw, const Bytes& encoded, const Expected& e) {
    bool ok = canonical(raw);
    for (std::size_t i = 0; i < 4; ++i) ok &= raw[i] == e.raw[i] && raw[i] == e.cpu_raw[i];
    for (std::size_t i = 0; i < 32; ++i) ok &= encoded[i] == e.encoded[i] && encoded[i] == e.cpu_encoded[i];
    return ok;
}
bool raw_matches(const Wide& raw, const std::array<std::uint8_t, 64>& encoded,
                 const Wide& expected, const std::array<std::uint8_t, 64>& expected_bytes) {
    bool ok = true;
    for (std::size_t i = 0; i < 8; ++i) ok &= raw[i] == expected[i];
    for (std::size_t i = 0; i < 64; ++i) ok &= encoded[i] == expected_bytes[i];
    return ok;
}
Expected reference(const cpp_int& expected, const FE& cpu, Counts& c) {
    const auto before = cpu.limbs();
    Expected out{words<4>(expected), bytes<32>(expected), before, cpu.to_bytes()};
    require(field_matches(out.cpu_raw, out.cpu_encoded, out), c,
            "CPU reference mismatch expected=" + hex(out.encoded) + " actual=" + hex(serialize(before)));
    require(cpu.limbs() == before, c, "CPU serialization mutation");
    ++c.cpu_checks;
    return out;
}
FE cpu_input(const cpp_int& x, Counts& c) {
    const FE out = FE::from_bytes(bytes<32>(x));
    require(out.limbs() == words<4>(x) && out.to_bytes() == bytes<32>(x), c, "CPU input construction");
    return out;
}
std::uint64_t random64(std::uint64_t& state) {
    std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
cpp_int random256(std::uint64_t& state) {
    Limbs x{}; for (auto& v : x) v = random64(state); return integer(x);
}

// Guards surround host allocations only. The test does not inspect device
// allocations and does not claim cudaMemcpy canaries detect device overruns.
constexpr std::uint64_t kGuard = UINT64_C(0xfedcba9876543210);
constexpr std::uint64_t kPoison = UINT64_C(0xdeadbeef55aa9966);
struct GuardedFields {
    std::size_t count;
    std::vector<P7Field> storage;
    explicit GuardedFields(std::size_t n) : count(n), storage(n + 4, field(Limbs{kGuard,kGuard,kGuard,kGuard})) {
        for (std::size_t i = 0; i < n; ++i) storage[i + 2] = field(Limbs{kPoison,kPoison,kPoison,kPoison});
    }
    explicit GuardedFields(const std::vector<P7Field>& x) : GuardedFields(x.size()) {
        for (std::size_t i = 0; i < count; ++i) storage[i + 2] = x[i];
    }
    P7Field* data() { return storage.data() + 2; }
    bool guards() const {
        for (auto i : {std::size_t(0), std::size_t(1), count + 2, count + 3})
            for (auto v : storage[i].limbs) if (v != kGuard) return false;
        return true;
    }
};
struct GuardedRaw {
    std::size_t count;
    std::vector<std::uint64_t> storage;
    explicit GuardedRaw(std::size_t n) : count(n), storage(n * 8 + 16, kGuard) {
        for (std::size_t i = 0; i < n * 8; ++i) storage[i + 8] = kPoison;
    }
    std::uint64_t* data() { return storage.data() + 8; }
    bool guards() const {
        for (std::size_t i = 0; i < 8; ++i)
            if (storage[i] != kGuard || storage[count * 8 + 8 + i] != kGuard) return false;
        return true;
    }
};
template<class T> bool unchanged(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
std::string witness(unsigned route, const std::string& op, unsigned steps, std::size_t index,
                    const P7Field& a, const P7Field& b) {
    return "route=" + std::to_string(route) + " op=" + op + " steps=" + std::to_string(steps) +
           " index=" + std::to_string(index) + " a=" + hex(serialize(limbs(a))) + " b=" + hex(serialize(limbs(b)));
}
void success(int rc, Counts& c, const std::string& context) {
    const std::string error = pa_p7_error();
    require(rc == 0, c, context + " API error=" + error);
    require(error.empty(), c, context + " success did not clear API error");
}
std::vector<P7Field> eval(unsigned route, const std::vector<P7Field>& a, const std::vector<P7Field>& b,
                          unsigned steps, bool alias, const std::vector<Expected>& expected,
                          Counts& c, const std::string& label) {
    require(a.size() == b.size() && a.size() == expected.size(), c, "eval fixture lengths");
    GuardedFields ga(a), gb(b), out(a.size());
    const auto before_a = ga.storage, before_b = gb.storage;
    ++c.eval_calls;
    success(pa_p7_eval(route, ga.data(), alias ? ga.data() : gb.data(), out.data(), a.size(), steps), c, label);
    require(unchanged(ga.storage, before_a) && unchanged(gb.storage, before_b), c, label + " input/canary mutation");
    require(out.guards(), c, label + " host output canary mutation");
    ++c.preservation_checks; ++c.canary_checks;
    std::vector<P7Field> result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = out.data()[i];
        const auto raw = limbs(result[i]); const auto encoded = serialize(raw);
        if (!field_matches(raw, encoded, expected[i]))
            require(false, c, witness(route, label, steps, i, a[i], alias ? a[i] : b[i]) +
                    " actual=" + hex(encoded) + " expected=" + hex(expected[i].encoded) +
                    " cpu=" + hex(expected[i].cpu_encoded) + " canonical=" + (canonical(raw) ? "true" : "false"));
        ++c.assertions; ++c.field_outputs; ++c.field_by_route[route]; mix(encoded, c);
    }
    return result;
}
void raw_eval(unsigned route, const std::vector<P7Field>& a, const std::vector<P7Field>& b,
              bool alias, Counts& c) {
    GuardedFields ga(a), gb(b); GuardedRaw out(a.size());
    const auto before_a = ga.storage, before_b = gb.storage;
    ++c.raw_calls;
    success(pa_p7_raw(route, ga.data(), alias ? ga.data() : gb.data(), out.data(), a.size()), c, "raw");
    require(unchanged(ga.storage, before_a) && unchanged(gb.storage, before_b), c, "raw input/canary mutation");
    require(out.guards(), c, "raw host output canary mutation");
    ++c.preservation_checks; ++c.canary_checks;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const cpp_int expected = integer(limbs(a[i])) * integer(limbs(alias ? a[i] : b[i]));
        Wide actual{}; for (std::size_t j = 0; j < 8; ++j) actual[j] = out.data()[i * 8 + j];
        const auto encoded = serialize(actual), expected_bytes = bytes<64>(expected);
        if (!raw_matches(actual, encoded, words<8>(expected), expected_bytes))
            require(false, c, witness(route, alias ? "raw_a_eq_b" : "raw", 1, i, a[i], alias ? a[i] : b[i]) +
                    " actual=" + hex(encoded) + " expected=" + hex(expected_bytes));
        ++c.assertions; ++c.raw_outputs; ++c.raw_by_route[route]; mix(encoded, c);
    }
}

void controls(Counts& c) {
    const cpp_int known("0x0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    const Limbs known_words{UINT64_C(0x191a1b1c1d1e1f20), UINT64_C(0x1112131415161718),
                            UINT64_C(0x090a0b0c0d0e0f10), UINT64_C(0x0102030405060708)};
    require(words<4>(known) == known_words && serialize(known_words) == bytes<32>(known), c, "known endian fixture");
    Expected e{words<4>(known), bytes<32>(known), words<4>(known), bytes<32>(known)};
    require(field_matches(e.raw, e.encoded, e), c, "positive field observer control");
    for (std::size_t i = 0; i < 32; ++i) {
        auto damaged = e.encoded; damaged[i] ^= 0x80;
        require(!field_matches(e.raw, damaged, e), c, "field byte observer blind spot"); ++c.negative_controls;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        auto damaged = e.raw; damaged[i] ^= UINT64_C(1) << 63;
        require(!field_matches(damaged, e.encoded, e), c, "field limb observer blind spot"); ++c.negative_controls;
    }
    Expected noncanonical{words<4>(kP), bytes<32>(kP), words<4>(kP), bytes<32>(kP)};
    require(!field_matches(noncanonical.raw, noncanonical.encoded, noncanonical), c, "noncanonical observer blind spot");
    ++c.negative_controls;
    const cpp_int product = (kB - 1) * (kB - 1);
    const Wide allmax_expected{1,0,0,0,UINT64_MAX-1,UINT64_MAX,UINT64_MAX,UINT64_MAX};
    require(words<8>(product) == allmax_expected && serialize(allmax_expected) == bytes<64>(product), c, "allmax full-product KAT");
    const auto wb = bytes<64>(product);
    require(raw_matches(allmax_expected, wb, allmax_expected, wb), c, "positive raw observer control");
    for (std::size_t i = 0; i < 64; ++i) {
        auto damaged = wb; damaged[i] ^= 0x80;
        require(!raw_matches(allmax_expected, damaged, allmax_expected, wb), c, "raw byte observer blind spot"); ++c.negative_controls;
    }
    for (std::size_t i = 0; i < 8; ++i) {
        auto damaged = allmax_expected; damaged[i] ^= UINT64_C(1) << 63;
        require(!raw_matches(damaged, wb, allmax_expected, wb), c, "raw limb observer blind spot"); ++c.negative_controls;
    }
    GuardedFields gf(1); GuardedRaw gr(1);
    require(gf.guards() && gr.guards(), c, "positive guard controls");
    gf.storage.front().limbs[3] ^= 1; gr.storage.back() ^= 1;
    require(!gf.guards() && !gr.guards(), c, "host guard observer blind spot"); c.negative_controls += 2;
}
void error_cases(Counts& c) {
    GuardedFields a(1), b(1), out(1); GuardedRaw raw(1);
    const auto sa = a.storage, sb = b.storage, so = out.storage; const auto sr = raw.storage;
    const auto reject = [&](int rc, const std::string& label) {
        const std::string error = pa_p7_error();
        require(rc == 1 && !error.empty(), c, label + " must fail with nonempty error");
        require(unchanged(a.storage, sa) && unchanged(b.storage, sb) && unchanged(out.storage, so) &&
                raw.storage == sr, c, label + " rejected API mutated host storage");
        ++c.error_rejections; ++c.preservation_checks;
    };
    for (unsigned route : {4U, std::numeric_limits<unsigned>::max()}) {
        reject(pa_p7_eval(route, a.data(), b.data(), out.data(), 1, 1), "eval invalid route");
        reject(pa_p7_raw(route, a.data(), b.data(), raw.data(), 1), "raw invalid route");
    }
    reject(pa_p7_eval(0, nullptr, b.data(), out.data(), 1, 1), "eval null a");
    reject(pa_p7_eval(0, a.data(), nullptr, out.data(), 1, 1), "eval null b");
    reject(pa_p7_eval(0, a.data(), b.data(), nullptr, 1, 1), "eval null out");
    reject(pa_p7_raw(0, nullptr, b.data(), raw.data(), 1), "raw null a");
    reject(pa_p7_raw(0, a.data(), nullptr, raw.data(), 1), "raw null b");
    reject(pa_p7_raw(0, a.data(), b.data(), nullptr, 1), "raw null out");
    for (std::size_t count : {std::size_t(0), P7_MAX_COUNT + 1, std::numeric_limits<std::size_t>::max()}) {
        reject(pa_p7_eval(0, a.data(), b.data(), out.data(), count, 1), "eval invalid count");
        reject(pa_p7_raw(0, a.data(), b.data(), raw.data(), count), "raw invalid count");
    }
    for (unsigned steps : {0U, P7_MAX_STEPS + 1, std::numeric_limits<unsigned>::max()})
        reject(pa_p7_eval(0, a.data(), b.data(), out.data(), 1, steps), "eval invalid steps");
}
struct Corpus {
    std::vector<P7Field> a, b;
    std::vector<Expected> product, square;
    std::set<std::pair<Limbs, Limbs>> distinct;
};
void append(Corpus& out, cpp_int a, cpp_int b, bool is_field, Counts& c) {
    if (is_field) { a = mod(a); b = mod(b); }
    out.a.push_back(field(a)); out.b.push_back(field(b));
    out.distinct.emplace(words<4>(a), words<4>(b));
    if (is_field) {
        const auto fa = cpu_input(a, c), fb = cpu_input(b, c);
        out.product.push_back(reference(mod(a * b), fa * fb, c));
        out.square.push_back(reference(mod(a * a), fa.square(), c));
    }
}
void corpus(Corpus& fields, Corpus& raw, Counts& c) {
    std::vector<cpp_int> edge{0,1,2,kP-2,kP-1,kP,kP+1,kB-2,kB-1};
    for (unsigned bit = 0; bit < 256; ++bit) {
        const cpp_int x = cpp_int(1) << bit;
        edge.push_back(x - 1); edge.push_back(x); edge.push_back(x + 1); edge.push_back(kB - 1 - x);
    }
    for (const Limbs& x : {Limbs{UINT64_C(0xaaaaaaaaaaaaaaaa),UINT64_C(0x5555555555555555),UINT64_C(0xaaaaaaaaaaaaaaaa),UINT64_C(0x5555555555555555)},
                           Limbs{UINT64_MAX,0,UINT64_MAX,0}, Limbs{0,UINT64_MAX,0,UINT64_MAX},
                           Limbs{UINT64_C(0xffffffff00000000),UINT64_C(0x00000000ffffffff),UINT64_C(0xffffffff00000000),UINT64_C(0x00000000ffffffff)}})
        edge.push_back(integer(x));
    for (std::size_t i = 0; i < edge.size(); ++i) {
        const cpp_int partner = i % 3 == 0 ? kB - 1 : i % 3 == 1 ? edge[(i + 1) % edge.size()] : cpp_int(1);
        append(fields, edge[i], partner, true, c); append(raw, edge[i], partner, false, c);
    }
    std::uint64_t state = kSeed;
    for (std::size_t i = 0; i < kRandomPairs; ++i) {
        const cpp_int a = random256(state), b = random256(state);
        append(fields, a, b, true, c); append(raw, a, b, false, c);
    }
    append(raw, kB - 1, kB - 1, false, c);
    const cpp_int a1 = kB - (cpp_int(1) << 33) - 1;
    const cpp_int a2 = (cpp_int(1) << 255) + 1, a3 = (cpp_int(1) << 255) - 1;
    require(mod(a1 * a1) == cpp_int("0xfffff860000e8900"), c, "P1 square carry KAT oracle");
    require(mod(a2 * (kP - 1)) == integer(Limbs{UINT64_C(0xfffffffefffffc2e),UINT64_MAX,UINT64_MAX,UINT64_C(0x7fffffffffffffff)}), c, "P1 multiply carry KAT oracle");
    require(mod(a3 * a3) == cpp_int("0x400000000000000000000000000000000000000000000000400001e740039f64"), c, "older large square KAT oracle");
    append(fields, a1, a1, true, c); append(fields, a2, kP - 1, true, c); append(fields, a3, a3, true, c);
}
struct Chains {
    std::vector<P7Field> a, b;
    std::vector<std::vector<Expected>> prefix{ kChainLength + 1 };
    std::vector<Expected> alias_seven;
};
Chains chains(Counts& c) {
    Chains out; std::uint64_t state = kSeed ^ UINT64_C(0x434841494e);
    for (std::size_t lane = 0; lane < kChainLanes; ++lane) {
        cpp_int a = mod(random256(state)), b = mod(random256(state));
        if (lane == 0) { a = 0; b = kP - 1; }
        if (lane == 1) { a = kP - 1; b = 0; }
        if (lane == 2) { a = kP - 1; b = 1; }
        if (lane == 3) { a = kP - 1; b = kP - 1; }
        if (lane == 4) { a = kB - (cpp_int(1) << 33) - 1; b = a; }
        out.a.push_back(field(a)); out.b.push_back(field(b));
        cpp_int x = a; FE cpu = cpu_input(a, c); const FE rhs = cpu_input(b, c);
        for (unsigned step = 1; step <= kChainLength; ++step) {
            x = mod(x * b); cpu = cpu * rhs; out.prefix[step].push_back(reference(x, cpu, c));
        }
        x = a; cpu = cpu_input(a, c); const FE fixed = cpu;
        for (unsigned step = 0; step < 7; ++step) { x = mod(x * a); cpu = cpu * fixed; }
        out.alias_seven.push_back(reference(x, cpu, c));
    }
    return out;
}
std::string json_string(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(ch);
        else out << ch;
    }
    out << '"'; return out.str();
}
} // namespace

int main(int argc, char** argv) {
    Counts c;
    try {
        int selected = -1;
        if (argc == 3 && std::string(argv[1]) == "--route") {
            const std::string value = argv[2];
            if (value.size() != 1 || value[0] < '0' || value[0] > '3') throw std::runtime_error("--route requires 0, 1, 2, or 3");
            selected = value[0] - '0';
        } else if (argc != 1) throw std::runtime_error("usage: test_p7_cuda_views [--route 0|1|2|3]");
        controls(c); error_cases(c); // All error cases are rejected before CUDA.
        Corpus fields, raw; corpus(fields, raw, c); const Chains trace = chains(c);
        for (unsigned route = 0; route < 4; ++route) {
            if (selected >= 0 && route != static_cast<unsigned>(selected)) continue;
            raw_eval(route, raw.a, raw.b, false, c); raw_eval(route, raw.a, raw.b, true, c);
            eval(route, fields.a, fields.b, 1, false, fields.product, c, "mul");
            eval(route, fields.a, fields.b, 1, true, fields.square, c, "mul_a_eq_b");
            for (unsigned step : {1U,2U,3U,7U,16U,31U,64U,127U,256U})
                eval(route, trace.a, trace.b, step, false, trace.prefix[step], c, "chain_prefix");
            auto state = trace.a;
            for (unsigned step = 1; step <= 8; ++step)
                state = eval(route, state, trace.b, 1, false, trace.prefix[step], c, "chain_roundtrip_" + std::to_string(step));
            // a==b means fixed ORIGINAL RHS. Seven multiplications yield a^8,
            // not repeated squaring; this checks alias semantics across steps.
            eval(route, trace.a, trace.a, 7, true, trace.alias_seven, c, "chain_a_eq_b");
        }
        std::ostringstream checksum; checksum << std::hex << std::setfill('0') << std::setw(16) << c.checksum;
        std::cout << "{\"status\":\"pass\",\"scope\":\"finite_P7_eval_raw_only\",\"seed\":\"5041375f56494557\",\"selected_route\":" << selected
                  << ",\"assertions\":" << c.assertions << ",\"field_pair_entries\":" << fields.a.size()
                  << ",\"field_distinct_pairs\":" << fields.distinct.size() << ",\"raw_pair_entries\":" << raw.a.size()
                  << ",\"raw_distinct_pairs\":" << raw.distinct.size() << ",\"random_pairs_per_domain\":" << kRandomPairs
                  << ",\"cpu_reference_checks\":" << c.cpu_checks << ",\"field_outputs\":" << c.field_outputs
                  << ",\"raw_outputs\":" << c.raw_outputs << ",\"eval_calls\":" << c.eval_calls << ",\"raw_calls\":" << c.raw_calls
                  << ",\"chain_lanes\":" << kChainLanes << ",\"chain_max_steps\":" << kChainLength
                  << ",\"prefixes_per_route\":9,\"roundtrip_steps_per_route\":8,\"alias_fixed_rhs_steps\":7"
                  << ",\"error_rejections\":" << c.error_rejections << ",\"negative_controls\":" << c.negative_controls
                  << ",\"input_preservation_checks\":" << c.preservation_checks << ",\"host_output_canary_checks\":" << c.canary_checks
                  << ",\"field_by_route\":[" << c.field_by_route[0] << ',' << c.field_by_route[1] << ',' << c.field_by_route[2] << ',' << c.field_by_route[3]
                  << "],\"raw_by_route\":[" << c.raw_by_route[0] << ',' << c.raw_by_route[1] << ',' << c.raw_by_route[2] << ',' << c.raw_by_route[3]
                  << "],\"host_canaries_only\":true,\"device_memory_sanitizer_claim\":false,\"timing_claim\":false,\"ct_claim\":false"
                  << ",\"checksum\":\"" << checksum.str() << "\"}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"status\":\"fail\",\"seed\":\"5041375f56494557\",\"assertions\":" << c.assertions
                  << ",\"field_outputs_checked\":" << c.field_outputs << ",\"raw_outputs_checked\":" << c.raw_outputs
                  << ",\"error\":" << json_string(e.what()) << "}\n";
        return 1;
    }
}
