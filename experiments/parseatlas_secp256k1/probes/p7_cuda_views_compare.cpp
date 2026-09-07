#include "p7_cuda_views.hpp"
#include "secp256k1/field.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using FE = secp256k1::fast::FieldElement;
constexpr unsigned steps = 128;
constexpr unsigned warmups = 2;
constexpr unsigned max_attempts = 24;
constexpr unsigned max_launches = 65536;
constexpr std::uint64_t max_region_ops = UINT64_C(1) << 34;
constexpr std::array<std::size_t,2> counts{32,32768};
constexpr std::array<const char*,4> names{
    "production_hybrid", "clone_explicit_shifts", "clone_memcpy_view", "existing_all64_contrast"};
struct Options {
    std::uint64_t seed = 20260905;
    double min_ms = 200;
    bool smoke = false;
    bool reverse = false;
};
struct Corpus { std::vector<P7Field> a,b,expected; };
struct Cell {
    unsigned route = 0;
    std::size_t count = 0;
    unsigned launches = 1;
    unsigned consecutive = 0;
    bool calibrated = false;
    std::vector<double> ns_per_op;
};
struct Raw {
    std::string phase;
    unsigned round;
    unsigned attempt;
    std::size_t sequence;
    unsigned cell;
    unsigned launches;
    std::uint64_t operations;
    double elapsed_ms;
    bool duration_qualified;
    bool output_checked;
    std::uint64_t checksum;
    std::string api_error;
};

std::string quoted(std::string_view value) {
    std::string out = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) {
            out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
        } else out += char(c);
    }
    return out + '"';
}
std::string hex64(std::uint64_t value) {
    std::string out(16,'0');
    constexpr char digits[] = "0123456789abcdef";
    for (int i=15; i>=0; --i) { out[std::size_t(i)] = digits[value & 15]; value >>= 4; }
    return out;
}
std::uint64_t integer(std::string_view text) {
    std::uint64_t value = 0;
    const auto [end,error] = std::from_chars(text.data(),text.data()+text.size(),value);
    if (error != std::errc{} || end != text.data()+text.size() || text.empty())
        throw std::invalid_argument("invalid unsigned integer");
    return value;
}
Options options(int argc, char** argv) {
    Options out;
    bool explicit_ms = false;
    for (int i=1; i<argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--smoke") out.smoke = true;
        else if (arg == "--reverse-order") out.reverse = true;
        else if (arg == "--seed" || arg == "--min-ms") {
            if (++i == argc) throw std::invalid_argument("missing option value");
            const auto n = integer(argv[i]);
            if (arg == "--seed") out.seed = n;
            else {
                if (n < 1 || n > 2000) throw std::invalid_argument("min-ms must be in [1,2000]");
                out.min_ms = double(n); explicit_ms = true;
            }
        } else throw std::invalid_argument("unknown option: " + std::string(arg));
    }
    if (out.smoke && !explicit_ms) out.min_ms = 1;
    if (!out.smoke && out.min_ms < 200) throw std::invalid_argument("full-series min-ms must be >=200");
    return out;
}
std::uint64_t splitmix(std::uint64_t& state) {
    std::uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
bool canonical(const P7Field& x) {
    return x.limbs[3] != UINT64_MAX || x.limbs[2] != UINT64_MAX ||
           x.limbs[1] != UINT64_MAX || x.limbs[0] < UINT64_C(0xfffffffefffffc2f);
}
P7Field random_field(std::uint64_t& state) {
    P7Field out{};
    do { for (auto& limb : out.limbs) limb = splitmix(state); }
    while (!canonical(out) || !(out.limbs[0]|out.limbs[1]|out.limbs[2]|out.limbs[3]));
    return out;
}
FE to_cpu(const P7Field& x) {
    return FE::from_limbs_raw({x.limbs[0],x.limbs[1],x.limbs[2],x.limbs[3]});
}
Corpus make_corpus(std::uint64_t seed) {
    Corpus result;
    result.a.resize(counts.back()); result.b.resize(counts.back());
    result.expected.resize(counts.back());
    for (std::size_t i=0; i<counts.back(); ++i) {
        result.a[i] = random_field(seed); result.b[i] = random_field(seed);
        FE x = to_cpu(result.a[i]);
        const FE rhs = to_cpu(result.b[i]);
        for (unsigned j=0; j<steps; ++j) x = x * rhs;
        for (unsigned limb=0; limb<4; ++limb) result.expected[i].limbs[limb] = x.limbs()[limb];
        if (!canonical(result.expected[i])) throw std::runtime_error("CPU reference returned noncanonical limbs");
    }
    return result;
}
std::uint64_t checksum(const P7Field* out, std::size_t count) {
    // Explicit little-limb byte order hashes every output byte, independent of
    // host struct padding (P7Field is exactly32 bytes).
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (std::size_t i=0; i<count; ++i)
        for (std::uint64_t limb : out[i].limbs)
            for (unsigned byte=0; byte<8; ++byte) {
                hash = (hash ^ (limb & 255)) * UINT64_C(1099511628211); limb >>= 8;
            }
    return hash;
}
bool matches(const std::vector<P7Field>& out, const Corpus& corpus, std::size_t count) {
    for (std::size_t i=0; i<count; ++i) {
        if (!canonical(out[i])) return false;
        for (unsigned limb=0; limb<4; ++limb)
            if (out[i].limbs[limb] != corpus.expected[i].limbs[limb]) return false;
    }
    return true;
}
std::array<unsigned,8> ordering(unsigned round, bool reverse) {
    std::array<unsigned,8> out{};
    for (unsigned i=0; i<8; ++i) out[i] = (i + round) % 8;
    if ((round & 1U) != unsigned(reverse)) std::reverse(out.begin(),out.end());
    return out;
}
unsigned grow(unsigned previous, double elapsed, double target, std::size_t count) {
    if (!(elapsed > 0) || !std::isfinite(elapsed)) throw std::runtime_error("nonpositive/nonfinite CUDA event time");
    const double estimate = std::ceil(double(previous) * std::max(1.5,1.25*target/elapsed));
    const auto work_cap = max_region_ops/(std::uint64_t(count)*steps);
    const auto cap = std::min<std::uint64_t>(max_launches,work_cap);
    if (estimate > double(cap) || previous >= cap)
        throw std::runtime_error("duration qualification exceeded launch/work cap");
    return std::max(previous+1,unsigned(estimate));
}
Raw measure(unsigned id, Cell& cell, const Corpus& corpus, const Options& opts,
            std::string phase, unsigned round, unsigned attempt, std::size_t sequence,
            std::vector<P7Field>& output) {
    float elapsed = 0;
    const int status = pa_p7_bench(cell.route,corpus.a.data(),corpus.b.data(),output.data(),
                                   cell.count,steps,cell.launches,&elapsed);
    const std::string error = status ? pa_p7_error() : "";
    const bool valid = !status && matches(output,corpus,cell.count);
    return {std::move(phase),round,attempt,sequence,id,cell.launches,
            std::uint64_t(cell.count)*steps*cell.launches,double(elapsed),
            !status && std::isfinite(elapsed) && double(elapsed) >= opts.min_ms,
            valid,status ? 0 : checksum(output.data(),cell.count),error};
}
void kernel_info(std::ostream& out, const P7KernelInfo& info) {
    out << "{\"registers\":" << info.registers
        << ",\"local_bytes\":" << info.local_bytes
        << ",\"shared_bytes\":" << info.shared_bytes
        << ",\"max_threads_per_block\":" << info.max_threads_per_block
        << ",\"binary_version\":" << info.binary_version
        << ",\"ptx_version\":" << info.ptx_version << '}';
}
void emit(const Options& opts, const P7Info& info, const std::array<Cell,8>& cells,
          const std::vector<Raw>& raw, const std::string& failure,
          double setup_ms, std::int64_t start_ms, std::int64_t end_ms,
          std::uint64_t corpus_a_hash, std::uint64_t corpus_b_hash) {
    auto& out = std::cout;
    out << std::setprecision(17) << std::boolalpha;
    out << "{\"schema\":\"parseatlas_p7_cuda_views_v1\",\"qualified\":" << failure.empty()
        << ",\"failure\":" << quoted(failure) << ",\"smoke_only\":" << opts.smoke
        << ",\"performance_eligible\":" << (!opts.smoke && failure.empty())
        << ",\"seed\":" << opts.seed << ",\"reverse_order\":" << opts.reverse
        << ",\"min_ms\":" << opts.min_ms << ",\"steps_per_thread\":" << steps
        << ",\"warmup_rounds\":2,\"measured_rounds\":" << (opts.smoke ? 2 : 8)
        << ",\"block_threads\":128,\"start_unix_ms\":" << start_ms << ",\"end_unix_ms\":" << end_ms
        << ",\"cpu_corpus_reference_setup_ms\":" << setup_ms
        << ",\"corpus_a_fnv1a64\":" << quoted(hex64(corpus_a_hash))
        << ",\"corpus_b_fnv1a64\":" << quoted(hex64(corpus_b_hash))
        << ",\"host_compiler\":" << quoted(__VERSION__)
        << ",\"contract\":{\"timing\":\"one-stream CUDA begin/end events around fixed-count repeated kernel launches; allocation, H2D/D2H, device endian check, JIT admission and one full untimed warm launch per API call excluded; event interval can include stream idle time from host launch submission\","
           "\"dependency\":\"each thread starts from a[i], evolves x=mul(x,b[i])128 times; each launch resets original seed; no inter-thread or per-step barrier\","
           "\"duration\":\"two consecutive qualifying calibration intervals at unchanged launch count; each warmup/measurement short interval is retained and retried with bounded larger count; only its final qualifying interval enters summary, never summed across stopped intervals\","
           "\"ordering\":\"eight cells rotated by round and alternately reversed; reverse-order flips each order; calibration has same sweep rule\","
           "\"validation\":\"every returned last-launch output checked outside events against unchanged corrected native CPU FE multiplication, all32bytes and raw canonicality; identical earlier launches not copied; independent raw-product/Boost oracle is separate gate\","
           "\"corpus\":\"runtime splitmix64 canonical nonzero a,b;32-thread workload is exact prefix of32768-thread corpus; RHS remains fixed through chain\","
           "\"comparison\":\"routes1/2 differ only in legal input loader; route3 changes product and reducer and is not view-only; count32 has one active warp, count32768 exposes many independent threads; ns/op is event-amortized throughput, not instruction latency\","
           "\"claims\":\"no constant-time certification, no owner-private pointer variant replication, no GPU clock lock or noise-isolation claim\"}"
        << ",\"limits\":{\"attempts\":" << max_attempts << ",\"launches\":" << max_launches
        << ",\"region_operations\":" << max_region_ops << "}"
        << ",\"device\":{\"name\":" << quoted(info.name) << ",\"ordinal\":" << info.device
        << ",\"major\":" << info.major << ",\"minor\":" << info.minor
        << ",\"multiprocessors\":" << info.multiprocessors
        << ",\"driver_version\":" << info.driver_version << ",\"runtime_version\":" << info.runtime_version
        << ",\"little_endian_checked\":" << bool(info.little_endian_checked)
        << ",\"hybrid_mul\":" << info.hybrid_mul << ",\"montgomery\":" << info.montgomery << "}"
        << ",\"routes\":[";
    for (unsigned route=0; route<4; ++route) {
        if (route) out << ',';
        out << "{\"route\":" << route << ",\"name\":" << quoted(names[route]) << ",\"field_attributes\":";
        kernel_info(out,info.field[route]); out << ",\"raw_attributes\":"; kernel_info(out,info.raw[route]); out << '}';
    }
    out << "],\"raw_regions\":[";
    for (std::size_t i=0; i<raw.size(); ++i) {
        if (i) out << ',';
        const auto& r = raw[i];
        out << "{\"sequence\":" << r.sequence << ",\"phase\":" << quoted(r.phase)
            << ",\"round\":" << r.round << ",\"attempt\":" << r.attempt << ",\"cell\":" << r.cell
            << ",\"route\":" << cells[r.cell].route << ",\"count\":" << cells[r.cell].count
            << ",\"launches\":" << r.launches << ",\"operations\":" << r.operations
            << ",\"elapsed_ms\":";
        if (std::isfinite(r.elapsed_ms)) out << r.elapsed_ms; else out << "null";
        out << ",\"duration_qualified\":" << r.duration_qualified << ",\"output_checked\":" << r.output_checked
            << ",\"checksum_fnv1a64\":" << quoted(hex64(r.checksum))
            << ",\"api_error\":" << quoted(r.api_error) << '}';
    }
    out << "],\"cells\":[";
    for (unsigned id=0; id<8; ++id) {
        if (id) out << ',';
        const auto& c = cells[id];
        auto values = c.ns_per_op;
        std::sort(values.begin(),values.end());
        out << "{\"cell\":" << id << ",\"route\":" << c.route << ",\"count\":" << c.count
            << ",\"calibrated\":" << c.calibrated << ",\"qualified_measurements\":" << values.size()
            << ",\"ns_per_op_samples\":[";
        for (std::size_t i=0; i<c.ns_per_op.size(); ++i) { if (i) out << ','; out << c.ns_per_op[i]; }
        out << ']';
        if (!values.empty()) {
            const auto middle = values.size()/2;
            const auto median = values.size()%2 ? values[middle] : (values[middle-1]+values[middle])/2;
            out << ",\"min_ns_per_op\":" << values.front() << ",\"median_ns_per_op\":" << median
                << ",\"max_ns_per_op\":" << values.back();
        }
        out << '}';
    }
    out << "]}\n";
}
std::int64_t unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: p7_cuda_views_compare [--seed UINT64] [--min-ms 200..2000] [--reverse-order] [--smoke]\n"
                     "JSON is emitted to stdout; redirect to a new file. Smoke defaults to1ms/two measured rounds and is never performance eligible.\n";
        return 0;
    }
    Options opts;
    try { opts = options(argc,argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    const auto start = unix_ms();
    P7Info info{};
    std::array<Cell,8> cells{};
    for (unsigned id=0; id<8; ++id) { cells[id].route = id%4; cells[id].count = counts[id/4]; }
    std::vector<Raw> raw;
    std::string failure;
    double setup_ms = 0;
    std::uint64_t a_hash = 0, b_hash = 0;
    try {
        if (pa_p7_info(&info)) throw std::runtime_error(pa_p7_error());
        const auto begin = std::chrono::steady_clock::now();
        const Corpus corpus = make_corpus(opts.seed);
        setup_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        a_hash = checksum(corpus.a.data(),corpus.a.size());
        b_hash = checksum(corpus.b.data(),corpus.b.size());
        std::vector<P7Field> output(counts.back());
        // Every timed API return is checked; failed outputs are retained before
        // stopping. There is no data-based/favorable-speed sample selection.
        auto run = [&](unsigned id, const char* phase, unsigned round, unsigned attempt) -> Raw {
            Raw r = measure(id,cells[id],corpus,opts,phase,round,attempt,raw.size(),output);
            raw.push_back(r);
            if (!r.api_error.empty()) throw std::runtime_error(r.api_error);
            if (!r.output_checked) throw std::runtime_error("GPU output differs from canonical CPU reference");
            if (!(r.elapsed_ms > 0) || !std::isfinite(r.elapsed_ms)) throw std::runtime_error("invalid CUDA event elapsed time");
            return r;
        };
        for (unsigned sweep=0; sweep<max_attempts; ++sweep) {
            for (unsigned id : ordering(sweep,opts.reverse)) {
                auto& c = cells[id];
                if (c.calibrated) continue;
                const auto r = run(id,"calibration",sweep,sweep);
                if (r.duration_qualified) {
                    if (++c.consecutive == 2) c.calibrated = true;
                } else {
                    c.consecutive = 0;
                    c.launches = grow(c.launches,r.elapsed_ms,opts.min_ms,c.count);
                }
            }
            if (std::all_of(cells.begin(),cells.end(),[](const Cell& c){ return c.calibrated; })) break;
        }
        if (!std::all_of(cells.begin(),cells.end(),[](const Cell& c){ return c.calibrated; }))
            throw std::runtime_error("calibration attempt cap exhausted");
        const unsigned rounds = opts.smoke ? 2 : 8;
        for (unsigned round=0; round<warmups+rounds; ++round) {
            for (unsigned id : ordering(round,opts.reverse)) {
                bool qualified = false;
                const bool warmup = round < warmups;
                for (unsigned attempt=0; attempt<max_attempts; ++attempt) {
                    const auto r = run(id,warmup ? "warmup" : "measurement",
                                       warmup ? round : round-warmups,attempt);
                    if (r.duration_qualified) {
                        if (!warmup) cells[id].ns_per_op.push_back(r.elapsed_ms*1e6/double(r.operations));
                        qualified = true; break;
                    }
                    cells[id].launches = grow(cells[id].launches,r.elapsed_ms,opts.min_ms,cells[id].count);
                }
                if (!qualified) throw std::runtime_error("warmup/measurement attempt cap exhausted");
            }
        }
    } catch (const std::exception& e) { failure = e.what(); }
    emit(opts,info,cells,raw,failure,setup_ms,start,unix_ms(),a_hash,b_hash);
    if (!failure.empty()) std::cerr << "P7 unqualified: " << failure << '\n';
    return failure.empty() ? 0 : 2;
}
