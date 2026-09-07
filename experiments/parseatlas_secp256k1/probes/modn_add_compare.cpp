// Native full-mod-n C++ comparison. Research only; production is unchanged.
#include "modn_add_control.hpp"
#include "secp256k1/scalar.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using pa_modn::Word;
using pa_modn::Limbs;
using pa_modn::Method;
using Scalar = secp256k1::fast::Scalar;
using State = std::array<Limbs,8>;
using ScalarState = std::array<Scalar,8>;
using Clock = std::chrono::steady_clock;
constexpr Word max_operations=1000000000, max_passes=1000000;
constexpr unsigned variant_count=4;
constexpr std::array<const char*,variant_count> variant_names{{
    "original", "blocked2", "gp_recomputed", "actual_scalar_api_diagnostic"}};
static_assert(sizeof(Limbs)==32 && sizeof(Scalar)==32, "RHS stride must match");

void require(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message); // Remains live under NDEBUG.
}
Word decimal(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("empty unsigned decimal");
    Word value=0;
    for (char ch:text) {
        if (ch<'0' || ch>'9') throw std::invalid_argument("invalid unsigned decimal");
        const Word digit=static_cast<Word>(ch-'0');
        if (value>(std::numeric_limits<Word>::max()-digit)/10)
            throw std::invalid_argument("unsigned decimal overflow");
        value=value*10+digit;
    }
    return value;
}
struct Options {
    bool smoke=false, throughput=false;
    unsigned cpu=4;
    Word seed=20260905, target_ms=200, count=256;
};
Options parse_options(int argc,char** argv) {
    if (argc<2 || (std::string(argv[1])!="--measure" && std::string(argv[1])!="--smoke"))
        throw std::invalid_argument("usage: modn_add_compare (--measure|--smoke) "
             "[--shape dependent|throughput] [--count 256|4096] [--cpu N] "
             "[--seed N] [--target-ms 200..1000]");
    Options result;
    result.smoke=std::string(argv[1])=="--smoke";
    bool shape=false,count=false,cpu=false,seed=false,target=false;
    for (int i=2;i<argc;i+=2) {
        if (i+1>=argc) throw std::invalid_argument("missing option value");
        const std::string key=argv[i], text=argv[i+1];
        if (key=="--shape" && !shape) {
            if (text!="dependent" && text!="throughput")
                throw std::invalid_argument("shape must be dependent or throughput");
            result.throughput=text=="throughput"; shape=true;
        } else if (key=="--count" && !count) {
            const Word value=decimal(text);
            if (value!=256 && value!=4096) throw std::invalid_argument("count must be 256 or 4096");
            result.count=value; count=true;
        } else if (key=="--cpu" && !cpu) {
            const Word value=decimal(text);
            if (value>=CPU_SETSIZE) throw std::invalid_argument("CPU exceeds CPU_SETSIZE");
            result.cpu=static_cast<unsigned>(value); cpu=true;
        } else if (key=="--seed" && !seed) { result.seed=decimal(text); seed=true;
        } else if (key=="--target-ms" && !target) {
            const Word value=decimal(text);
            if (value<200 || value>1000) throw std::invalid_argument("target must be 200..1000 ms");
            result.target_ms=value; target=true;
        } else throw std::invalid_argument("unknown or duplicate option: "+key);
    }
    if (result.smoke && target) throw std::invalid_argument("smoke has no performance target");
    return result;
}
bool valid_work(Word count,Word passes) {
    return (count==256 || count==4096) && passes>0 && passes<=max_passes &&
           passes<=max_operations/count;
}
std::vector<unsigned> pin_cpu(unsigned cpu) {
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (sched_getaffinity(0,sizeof(allowed),&allowed)!=0)
        throw std::runtime_error("sched_getaffinity failed");
    std::vector<unsigned> before;
    for (unsigned i=0;i<CPU_SETSIZE;++i) if (CPU_ISSET(i,&allowed)) before.push_back(i);
    if (!CPU_ISSET(cpu,&allowed)) throw std::invalid_argument("CPU outside inherited affinity");
    cpu_set_t selected; CPU_ZERO(&selected); CPU_SET(cpu,&selected);
    if (sched_setaffinity(0,sizeof(selected),&selected)!=0)
        throw std::runtime_error("sched_setaffinity failed");
    require(sched_getcpu()==static_cast<int>(cpu),"selected CPU not established");
    return before;
}
Word splitmix64(Word& state) {
    Word z=(state+=UINT64_C(0x9e3779b97f4a7c15));
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
    return z^(z>>31);
}
Limbs random_canonical(Word& seed) {
    for (unsigned trial=0;trial<1000;++trial) {
        Limbs value{};
        for (Word& word:value) word=splitmix64(seed);
        if (pa_modn::canonical(value)) return value; // Rejection, not biased masking.
    }
    throw std::runtime_error("canonical rejection cap reached");
}
struct Inputs {
    State initial{};
    ScalarState initial_api{};
    std::vector<Limbs> rhs;
    std::vector<Scalar> rhs_api;
};
Inputs make_inputs(Word seed,Word count) {
    Inputs result;
    result.rhs.reserve(static_cast<std::size_t>(count));
    result.rhs_api.reserve(static_cast<std::size_t>(count));
    for (std::size_t lane=0;lane<result.initial.size();++lane) {
        result.initial[lane]=random_canonical(seed);
        result.initial_api[lane]=Scalar::from_limbs(result.initial[lane]);
        require(pa_modn::encode_le(result.initial[lane])==
                pa_modn::encode_le(result.initial_api[lane].limbs()),"initial API conversion mismatch");
    }
    for (Word i=0;i<count;++i) {
        result.rhs.push_back(random_canonical(seed));
        result.rhs_api.push_back(Scalar::from_limbs(result.rhs.back()));
        require(pa_modn::encode_le(result.rhs.back())==
                pa_modn::encode_le(result.rhs_api.back().limbs()),"RHS API conversion mismatch");
    }
    return result;
}
State extract(const ScalarState& input) {
    State result{};
    for (std::size_t lane=0;lane<input.size();++lane) result[lane]=input[lane].limbs();
    return result;
}
bool same_state(const State& a,const State& b) {
    for (std::size_t lane=0;lane<a.size();++lane)
        if (pa_modn::encode_le(a[lane])!=pa_modn::encode_le(b[lane])) return false;
    return true;
}
Word checksum(const State& state) {
    Word result=UINT64_C(0xcbf29ce484222325);
    for (const auto& lane:state)
        for (std::uint8_t byte:pa_modn::encode_le(lane))
            result=(result^byte)*UINT64_C(0x100000001b3);
    return result;
}
template<unsigned V> using Value = std::conditional_t<V==3,Scalar,Limbs>;
template<unsigned V> using RegionState = std::array<Value<V>,8>;
template<unsigned V> inline void step(Value<V>& a,const Value<V>& b) {
    static_assert(V<variant_count,"unknown variant");
    if constexpr (V==3) a+=b; // Diagnostic intentionally retains the public API call.
    else pa_modn::add_assign<static_cast<Method>(V)>(a,b);
}
// Unchecked inner region: count/pass/range and canonical inputs are checked
// before dispatch. No cross-lane dependence, per-add barrier or runtime dispatch.
template<unsigned V,bool Throughput>
inline void advance(RegionState<V>& state,const Value<V>* rhs,Word count,Word passes) {
    RegionState<V> local=state;
    for (Word pass=0;pass<passes;++pass) {
        if constexpr (Throughput) {
            for (Word i=0;i<count;i+=8) {
                step<V>(local[0],rhs[i]);   step<V>(local[1],rhs[i+1]);
                step<V>(local[2],rhs[i+2]); step<V>(local[3],rhs[i+3]);
                step<V>(local[4],rhs[i+4]); step<V>(local[5],rhs[i+5]);
                step<V>(local[6],rhs[i+6]); step<V>(local[7],rhs[i+7]);
            }
        } else {
            for (Word i=0;i<count;++i) step<V>(local[0],rhs[i]);
        }
    }
    state=local;
}
inline void observe_memory(const void* pointer) {
    asm volatile("" : : "g"(pointer) : "memory");
}
struct RegionClock { Word start_ns,stop_ns,elapsed_ns; };
template<unsigned V,bool Throughput>
__attribute__((noinline)) RegionClock timed_region(RegionState<V>& state,
                                                  const Value<V>* rhs,Word count,Word passes) {
    observe_memory(&state);
    observe_memory(rhs);
    const auto start=Clock::now();
    // Same state copy and final writeback boundary for all matched inline methods.
    advance<V,Throughput>(state,rhs,count,passes);
    observe_memory(&state);
    const auto stop=Clock::now();
    const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
    require(elapsed>0,"nonpositive elapsed time");
    return {static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                start.time_since_epoch()).count()),
            static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                stop.time_since_epoch()).count()),static_cast<Word>(elapsed)};
}
struct Sample {
    unsigned variant,round,position;
    Word count,passes;
    RegionClock clock;
    State final_state;
    int cpu_before,cpu_after;
};
template<unsigned V,bool Throughput>
Sample run_variant(const Options& options,const Inputs& inputs,Word passes,
                   unsigned round,unsigned position) {
    require(valid_work(options.count,passes),"invalid timed work");
    require(inputs.rhs.size()==options.count && inputs.rhs_api.size()==options.count,
            "RHS range mismatch");
    RegionState<V> state;
    const Value<V>* rhs;
    if constexpr (V==3) { state=inputs.initial_api;rhs=inputs.rhs_api.data(); }
    else { state=inputs.initial;rhs=inputs.rhs.data(); }
    const int before=sched_getcpu();
    require(before==static_cast<int>(options.cpu),"CPU changed before sample");
    // Same first 256 operands, semantics and resulting initial state in all variants.
    advance<V,Throughput>(state,rhs,256,1);
    const auto timing=timed_region<V,Throughput>(state,rhs,options.count,passes);
    const int after=sched_getcpu();
    require(after==static_cast<int>(options.cpu),"CPU changed after sample");
    State final_state;
    if constexpr (V==3) final_state=extract(state);
    else final_state=state;
    return {V,round,position,options.count,passes,timing,final_state,before,after};
}
template<bool Throughput>
Sample dispatch_shape(unsigned variant,const Options& options,const Inputs& inputs,
                      Word passes,unsigned round,unsigned position) {
    switch (variant) {
        case 0:return run_variant<0,Throughput>(options,inputs,passes,round,position);
        case 1:return run_variant<1,Throughput>(options,inputs,passes,round,position);
        case 2:return run_variant<2,Throughput>(options,inputs,passes,round,position);
        case 3:return run_variant<3,Throughput>(options,inputs,passes,round,position);
    }
    throw std::logic_error("unknown variant dispatch");
}
Sample dispatch(unsigned variant,const Options& options,const Inputs& inputs,
                Word passes,unsigned round,unsigned position) {
    if (options.throughput)
        return dispatch_shape<true>(variant,options,inputs,passes,round,position);
    return dispatch_shape<false>(variant,options,inputs,passes,round,position);
}
struct PathCounts { Word no_reduction=0,high_carry_reduction=0,threshold_reduction=0; };
template<unsigned V>
Word validate_prefix(const Options& options,const Inputs& inputs,PathCounts& paths) {
    State candidate=inputs.initial;
    ScalarState reference=inputs.initial_api;
    for (Word i=0;i<options.count;++i) {
        const Word lane=options.throughput?i%8:0;
        if constexpr (V==0) {
            // Untimed prefix coverage only, not measured-trajectory path counts.
            const auto raw=pa_l0_native::compute<pa_l0_native::Method::Original>(
                {candidate[lane],inputs.rhs[i],Word{0}});
            if (raw.carry_out) ++paths.high_carry_reduction;
            else if (!pa_modn::canonical(raw.low)) ++paths.threshold_reduction;
            else ++paths.no_reduction;
        }
        step<V>(candidate[lane],inputs.rhs[i]);
        reference[lane]+=inputs.rhs_api[i];
        require(pa_modn::canonical(candidate[lane]),"noncanonical prefix result");
        require(same_state(candidate,extract(reference)),"preflight complete byte mismatch");
    }
    return options.count;
}
std::array<unsigned,4> initial_order(Word seed) {
    seed^=UINT64_C(0xd1b54a32d192ed03);
    std::array<unsigned,4> order{{0,1,2,3}};
    for (std::size_t i=order.size()-1;i>0;--i)
        std::swap(order[i],order[splitmix64(seed)%(i+1)]);
    return order;
}
struct Collected {
    Word count,passes,preflight_checks;
    PathCounts prefix_paths;
    State reference;
    std::array<unsigned,4> order;
    std::vector<Sample> calibration,warmups,samples;
};
Collected collect(const Options& options) {
    Collected result{};
    result.count=options.count;
    result.passes=options.smoke?2:1;
    result.order=initial_order(options.seed);
    const Inputs inputs=make_inputs(options.seed,options.count);
    result.preflight_checks=validate_prefix<0>(options,inputs,result.prefix_paths)+
                            validate_prefix<1>(options,inputs,result.prefix_paths)+
                            validate_prefix<2>(options,inputs,result.prefix_paths);
    if (!options.smoke) {
        const Word target=options.target_ms*1000000;
        const Word cap=std::min(max_passes,max_operations/options.count);
        bool reached=false;
        for (unsigned trial=0;trial<8;++trial) {
            const Sample sample=dispatch(0,options,inputs,result.passes,trial,0);
            result.calibration.push_back(sample);
            if (sample.clock.elapsed_ns>=target) { reached=true;break; }
            const long double proposed=std::ceil(static_cast<long double>(result.passes)*
                                                target*1.2L/sample.clock.elapsed_ns);
            const Word next=proposed>=cap?cap:static_cast<Word>(proposed);
            require(next>result.passes,"calibration target impossible within caps");
            result.passes=next;
        }
        require(reached,"calibration target not reached in eight trials");
    }
    require(valid_work(options.count,result.passes),"invalid calibrated work");
    ScalarState reference=inputs.initial_api;
    if (options.throughput) {
        advance<3,true>(reference,inputs.rhs_api.data(),256,1);
        advance<3,true>(reference,inputs.rhs_api.data(),options.count,result.passes);
    } else {
        advance<3,false>(reference,inputs.rhs_api.data(),256,1);
        advance<3,false>(reference,inputs.rhs_api.data(),options.count,result.passes);
    }
    result.reference=extract(reference);
    for (const auto& lane:result.reference) require(pa_modn::canonical(lane),"noncanonical replay");
    if (!result.calibration.empty())
        require(same_state(result.calibration.back().final_state,result.reference),
                "final calibration/API replay mismatch");
    // A single pinned measurement worker is deliberate: concurrent samples
    // would change the machine conditions and introduce self-interference.
    for (unsigned phase=0;phase<2;++phase) {
        const unsigned rounds=phase==0?2:8;
        for (unsigned round=0;round<rounds;++round) {
            for (unsigned position=0;position<variant_count;++position) {
                const unsigned variant=result.order[(position+round)%variant_count];
                Sample sample=dispatch(variant,options,inputs,result.passes,round,position);
                require(same_state(sample.final_state,result.reference),"sample/API replay byte mismatch");
                (phase==0?result.warmups:result.samples).push_back(sample);
            }
        }
    }
    return result;
}
double quantile(const std::vector<double>& sorted,double fraction) {
    require(!sorted.empty(),"empty quantile");
    const double position=(sorted.size()-1)*fraction;
    const auto lo=static_cast<std::size_t>(position);
    const auto hi=std::min(lo+1,sorted.size()-1);
    return sorted[lo]+(sorted[hi]-sorted[lo])*(position-lo);
}
struct Statistics { double median,minimum,maximum,mad,iqr; };
Statistics summarize(std::vector<double> values) {
    require(!values.empty(),"empty summary");
    for (double value:values) require(std::isfinite(value),"nonfinite summary value");
    std::sort(values.begin(),values.end());
    const double median=quantile(values,.5);
    std::vector<double> deviations;
    for (double value:values) deviations.push_back(std::abs(value-median));
    std::sort(deviations.begin(),deviations.end());
    return {median,values.front(),values.back(),quantile(deviations,.5),
            quantile(values,.75)-quantile(values,.25)};
}
void selftest() {
    const auto odd=summarize({7,1,4,3,6,2,5});
    require(odd.median==4 && odd.minimum==1 && odd.maximum==7 && odd.mad==2 && odd.iqr==3,
            "odd statistics fixture failed");
    const auto even=summarize({8,1,7,2,6,3,5,4});
    require(even.median==4.5 && even.mad==2 && even.iqr==3.5,"eight-sample fixture failed");
    const auto one=summarize({3});
    require(one.median==3 && one.mad==0 && one.iqr==0,"singleton fixture failed");
    require(decimal("0")==0 && decimal("18446744073709551615")==UINT64_MAX,"decimal fixture failed");
    for (const char* invalid:{"","-1","+1","1x"," 1","18446744073709551616"}) {
        bool rejected=false;
        try { static_cast<void>(decimal(invalid)); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid decimal accepted");
    }
    require(valid_work(256,1) && valid_work(4096,max_operations/4096),"valid work rejected");
    require(!valid_work(0,1) && !valid_work(255,1) && !valid_work(256,0) &&
            !valid_work(256,max_passes+1) && !valid_work(4096,max_operations/4096+1) &&
            !valid_work(UINT64_MAX,UINT64_MAX),"invalid or overflowing work accepted");
    State fixture{}; State corrupted=fixture; corrupted[7][3]=1;
    require(same_state(fixture,fixture) && !same_state(fixture,corrupted),"full-state comparator fixture");
    const auto order=initial_order(20260905);
    std::array<std::array<unsigned,4>,4> appearances{};
    for (unsigned round=0;round<8;++round)
        for (unsigned position=0;position<4;++position) ++appearances[order[(position+round)%4]][position];
    for (const auto& row:appearances) for (unsigned n:row) require(n==2,"unbalanced rotation");
}
std::string quote(const std::string& value) {
    std::ostringstream output;output<<'"';
    for (unsigned char ch:value) {
        if (ch=='"'||ch=='\\') output<<'\\'<<ch;
        else if (ch<32) output<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')
                            <<static_cast<unsigned>(ch)<<std::dec;
        else output<<ch;
    }
    output<<'"';return output.str();
}
std::string hex(Word value) {
    std::ostringstream output;output<<std::hex<<std::setw(16)<<std::setfill('0')<<value;return output.str();
}
double ns_per_operation(const Sample& sample) {
    return static_cast<double>(sample.clock.elapsed_ns)/(sample.count*sample.passes);
}
void print_statistics(const Statistics& s) {
    std::cout<<"{\"median\":"<<s.median<<",\"min\":"<<s.minimum<<",\"max\":"<<s.maximum
             <<",\"mad\":"<<s.mad<<",\"inclusive_iqr\":"<<s.iqr<<'}';
}
void print_values(const std::vector<double>& values) {
    std::cout<<'[';
    for (std::size_t i=0;i<values.size();++i) { if(i)std::cout<<',';std::cout<<values[i]; }
    std::cout<<']';
}
void print_samples(const std::vector<Sample>& samples,Word target) {
    std::cout<<'[';
    for (std::size_t i=0;i<samples.size();++i) {
        const auto& s=samples[i];
        if(i)std::cout<<',';
        std::cout<<"{\"variant\":"<<s.variant<<",\"method\":"<<quote(variant_names[s.variant])
                 <<",\"matched_inline_candidate\":"<<(s.variant<3?"true":"false")
                 <<",\"round\":"<<s.round<<",\"position\":"<<s.position
                 <<",\"count\":"<<s.count<<",\"passes\":"<<s.passes<<",\"operations\":"<<s.count*s.passes
                 <<",\"start_steady_ns\":"<<s.clock.start_ns<<",\"stop_steady_ns\":"<<s.clock.stop_ns
                 <<",\"elapsed_ns\":"<<s.clock.elapsed_ns<<",\"ns_per_modular_addition\":"<<ns_per_operation(s)
                 <<",\"checksum\":"<<quote(hex(checksum(s.final_state)))
                 <<",\"cpu_before\":"<<s.cpu_before<<",\"cpu_after\":"<<s.cpu_after
                 <<",\"below_requested_target\":";
        if(!target)std::cout<<"null";else std::cout<<(s.clock.elapsed_ns<target?"true":"false");
        std::cout<<'}';
    }
    std::cout<<']';
}
Word unix_now_ns() {
    return static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count());
}
void print_report(const Collected& result,const Options& options,const std::vector<unsigned>& allowed,
                  Word started,Word finished) {
    const Word target=options.smoke?0:options.target_ms*1000000;
    std::cout<<std::setprecision(17)<<"{\"protocol\":\"parseatlas_modn_native_compare_v1\",\"mode\":"
             <<quote(options.smoke?"smoke_not_performance_evidence":"measurement")
             <<",\"timing_claim\":"<<(options.smoke?"false":"true")
             <<",\"shape\":"<<quote(options.throughput?"throughput":"dependent")
             <<",\"active_lanes\":"<<(options.throughput?8:1)
             <<",\"started_unix_ns\":"<<started<<",\"finished_unix_ns\":"<<finished
             <<",\"compiler_version\":"<<quote(__VERSION__)<<",\"cplusplus\":"<<__cplusplus
             <<",\"seed\":"<<options.seed<<",\"cpu\":"<<options.cpu<<",\"inherited_allowed_cpus\":[";
    for (std::size_t i=0;i<allowed.size();++i) { if(i)std::cout<<',';std::cout<<allowed[i]; }
#ifdef SECP256K1_NO_INT128
    constexpr bool no_int128=true;
#else
    constexpr bool no_int128=false;
#endif
#ifdef __SIZEOF_INT128__
    constexpr bool int128_macro=true;
#else
    constexpr bool int128_macro=false;
#endif
    std::cout<<"],\"original_backend\":"<<quote(no_int128||!int128_macro?"portable_carry":"gcc_uint128")
             <<",\"no_int128_defined\":"<<(no_int128?"true":"false")
             <<",\"int128_macro_defined\":"<<(int128_macro?"true":"false")
             <<",\"boundary_state_bytes\":"<<sizeof(State)<<",\"rhs_bytes_per_representation\":"<<32*result.count
             <<",\"requested_target_ns\":"<<target<<",\"count\":"<<result.count<<",\"passes\":"<<result.passes
             <<",\"operations_per_sample\":"<<result.count*result.passes
             <<",\"warmup_operations_per_sample\":256,\"warmup_rounds\":2,\"sample_rounds\":8"
               ",\"measurement_workers\":1,\"statistics_parser_order_work_selftest\":\"passed\""
             <<",\"preflight_complete_state_checks\":"<<result.preflight_checks
             <<",\"original_preflight_only_reduction_paths\":{\"no_reduction\":"
             <<result.prefix_paths.no_reduction<<",\"high_carry_reduction\":"
             <<result.prefix_paths.high_carry_reduction<<",\"threshold_reduction_without_carry\":"
             <<result.prefix_paths.threshold_reduction<<'}'
             <<",\"every_warmup_and_measured_final_state_compared_exactly\":true"
               ",\"comparison\":\"all_eight_lanes_serialized_little_endian_256_bytes\""
               ",\"replay_reference\":\"unchanged_public_fast_Scalar_operator_plus_assign\""
               ",\"final_calibration_replayed\":"
             <<(result.calibration.empty()?"null":"true")<<",\"reference_checksum\":"
             <<quote(hex(checksum(result.reference)))<<",\"initial_variant_order\":[";
    for (std::size_t i=0;i<result.order.size();++i) { if(i)std::cout<<',';std::cout<<result.order[i]; }
    std::cout<<"],\"conditions\":["
      "\"Full addition modulo scalar order n, canonical inputs and outputs, common variable-time reduction\","
      "\"Three matched inline carry methods; actual Scalar+= is a separate diagnostic API-call workload\","
      "\"Count is additions per pass for both shapes; throughput stripes the same RHS stream over eight lanes\","
      "\"Timed region includes local state copy and full final writeback; no per-add construction or dispatch\","
      "\"Two RHS representations coexist; only selected representation is accessed in timed region\","
      "\"The 8 KiB and 128 KiB RHS sizes are not claims of PMU-confirmed cache residency or DRAM bandwidth\","
      "\"Original calibrates common work; every calibration, warmup, sample and below-target sample retained\","
      "\"Intermediate calibration trials are not replay-checked; final calibration and all later samples are\","
      "\"Eight seeded rotating measured rounds place each variant at every position twice\","
      "\"Complete final bytes checked against actual public Scalar replay; separate Boost gate is independent\","
      "\"No ratio against actual API is emitted as a carry-algebra speedup; descriptive paired ratios only\","
      "\"One pinned serial worker avoids self-interference; external load, sibling activity and thermals uncontrolled\","
      "\"Only this process affinity changes; no sudo or global setting changes\","
      "\"No Python, PMU counters, novelty, constant-time proof, production accept or whole-engine claim\"]"
      ",\"calibration\":";
    print_samples(result.calibration,target);std::cout<<",\"warmups\":";print_samples(result.warmups,target);
    std::cout<<",\"samples\":";print_samples(result.samples,target);std::cout<<",\"summaries\":[";
    for (unsigned variant=0;variant<variant_count;++variant) {
        std::vector<double> elapsed,paired;
        unsigned below=0;
        for (const auto& sample:result.samples) if(sample.variant==variant) {
            elapsed.push_back(ns_per_operation(sample));
            if (variant<3) {
                const auto found=std::find_if(result.samples.begin(),result.samples.end(),
                    [&](const Sample& s){return s.variant==0 && s.round==sample.round;});
                require(found!=result.samples.end(),"paired Original missing");
                paired.push_back(ns_per_operation(*found)/ns_per_operation(sample));
            }
            if(target && sample.clock.elapsed_ns<target)++below;
        }
        if(variant)std::cout<<',';
        std::cout<<"{\"variant\":"<<variant<<",\"method\":"<<quote(variant_names[variant])
                 <<",\"matched_inline_candidate\":"<<(variant<3?"true":"false")
                 <<",\"sample_count\":"<<elapsed.size()<<",\"below_target_count\":"<<below
                 <<",\"ns_per_modular_addition\":";print_statistics(summarize(elapsed));
        std::cout<<",\"paired_original_over_candidate\":";
        if(variant<3)print_statistics(summarize(paired));else std::cout<<"null";
        std::cout<<",\"ratio_by_round\":";
        if(variant<3)print_values(paired);else std::cout<<"null";
        std::cout<<'}';
    }
    std::cout<<"]}\n";
}
} // namespace

int main(int argc,char** argv) {
    try {
        const auto options=parse_options(argc,argv);
        selftest();const auto allowed=pin_cpu(options.cpu);
        const Word started=unix_now_ns();
        const auto result=collect(options);
        const Word finished=unix_now_ns();
        print_report(result,options,allowed,started,finished);
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"modn_add_compare: "<<error.what()<<'\n';return 2;
    }
}
