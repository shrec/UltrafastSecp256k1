// Native C++ crossed recurrence-control driver; research only, not production.
#include "l0_state_control.hpp"
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
#include <vector>

namespace {
using pa_l0_state::Input;
using pa_l0_state::Method;
using pa_l0_state::Form;
using pa_l0_state::Word;
using Clock = std::chrono::steady_clock;
constexpr Word checksum_initial = UINT64_C(0xcbf29ce484222325);
struct Variant { Method method; Form form; };
constexpr std::array<Variant,8> variants{{
    {Method::Original,Form::Aggregate}, {Method::GpMaterialized,Form::Aggregate},
    {Method::GpRecomputed,Form::Aggregate}, {Method::Blocked2,Form::Aggregate},
    {Method::Original,Form::ScalarLocal}, {Method::GpMaterialized,Form::ScalarLocal},
    {Method::GpRecomputed,Form::ScalarLocal}, {Method::Blocked2,Form::ScalarLocal}}};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message); // Live under NDEBUG.
}
const char* method_name(Method method) {
    switch (method) {
        case Method::Original: return "original";
        case Method::GpMaterialized: return "gp_materialized";
        case Method::GpRecomputed: return "gp_recomputed";
        case Method::Blocked2: return "blocked2";
    }
    throw std::logic_error("unknown method");
}
const char* form_name(Form form) {
    return form == Form::Aggregate ? "aggregate" : "scalar_local";
}
Word decimal(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("empty unsigned decimal");
    Word value=0;
    for (char ch : text) {
        if (ch<'0' || ch>'9') throw std::invalid_argument("invalid unsigned decimal");
        const Word digit=static_cast<Word>(ch-'0');
        if (value>(std::numeric_limits<Word>::max()-digit)/10)
            throw std::invalid_argument("unsigned decimal overflow");
        value=value*10+digit;
    }
    return value;
}
struct Options { bool smoke=false; unsigned cpu=4; Word seed=20260905; Word target_ms=200; };
Options parse_options(int argc, char** argv) {
    if (argc<2 || (std::string(argv[1])!="--measure" && std::string(argv[1])!="--smoke"))
        throw std::invalid_argument("usage: l0_state_compare (--measure|--smoke) "
                                    "[--cpu N] [--seed N] [--target-ms 200..1000]");
    Options result;
    result.smoke=std::string(argv[1])=="--smoke";
    bool cpu=false, seed=false, target=false;
    for (int i=2; i<argc; i+=2) {
        if (i+1>=argc) throw std::invalid_argument("missing option value");
        const std::string key=argv[i];
        const Word value=decimal(argv[i+1]);
        if (key=="--cpu" && !cpu) {
            if (value>=CPU_SETSIZE) throw std::invalid_argument("CPU exceeds CPU_SETSIZE");
            result.cpu=static_cast<unsigned>(value); cpu=true;
        } else if (key=="--seed" && !seed) { result.seed=value; seed=true;
        } else if (key=="--target-ms" && !target) {
            if (value<200 || value>1000) throw std::invalid_argument("target must be 200..1000 ms");
            result.target_ms=value; target=true;
        } else throw std::invalid_argument("unknown or duplicate option: "+key);
    }
    if (result.smoke && target) throw std::invalid_argument("smoke has no performance target");
    return result;
}
std::vector<unsigned> pin_cpu(unsigned cpu) {
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (sched_getaffinity(0,sizeof(allowed),&allowed)!=0)
        throw std::runtime_error("sched_getaffinity failed");
    std::vector<unsigned> before;
    for (unsigned i=0; i<CPU_SETSIZE; ++i) if (CPU_ISSET(i,&allowed)) before.push_back(i);
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
Input make_input(Word seed) {
    Input input{};
    for (Word& word:input.a) word=splitmix64(seed);
    for (Word& word:input.b) word=splitmix64(seed);
    input.carry_in=splitmix64(seed)&1;
    return input;
}
bool same_state(const Input& a,const Input& b) {
    return a.a==b.a && a.b==b.b && a.carry_in==b.carry_in;
}
Word checksum(const Input& state) {
    Word result=checksum_initial;
    for (Word word:state.a) result=(result^word)*UINT64_C(0x100000001b3);
    for (Word word:state.b) result=(result^word)*UINT64_C(0x100000001b3);
    return (result^state.carry_in)*UINT64_C(0x100000001b3);
}
// Compiler barriers only at region boundaries, not hardware fences or forced
// per-step materialization. No volatile arithmetic or inner-loop dispatch.
inline void observe_memory(const void* pointer) {
    asm volatile("" : : "g"(pointer) : "memory");
}
struct RegionClock { Word start_ns,stop_ns,elapsed_ns; };
template<Method M,Form F>
__attribute__((noinline)) RegionClock timed_region(Input& state,Word count,Word passes) {
    observe_memory(&state);
    const auto start=Clock::now();
    // Unpack, all steps and complete final state writeback are inside the clock.
    pa_l0_state::advance<M,F>(state,count,passes);
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
    Input final_state;
    int cpu_before,cpu_after;
};
template<Method M,Form F>
Sample run_variant(unsigned variant,const Options& options,Word count,Word passes,
                   unsigned round,unsigned position) {
    Input state=make_input(options.seed);
    require(count && passes && pa_l0_state::valid_region(state,count,passes),"invalid timed region");
    const int before=sched_getcpu();
    require(before==static_cast<int>(options.cpu),"CPU changed before sample");
    // Identical state, length and semantics for every variant. The untimed
    // warmup leaves the same conditioned initial state for the measured region.
    pa_l0_state::advance<M,F>(state,std::min<Word>(count,4096),1);
    const auto timing=timed_region<M,F>(state,count,passes);
    const int after=sched_getcpu();
    require(after==static_cast<int>(options.cpu),"CPU changed after sample");
    return {variant,round,position,count,passes,timing,state,before,after};
}
Sample dispatch(unsigned variant,const Options& options,Word count,Word passes,
                unsigned round,unsigned position) {
    switch (variant) {
        case 0:return run_variant<Method::Original,Form::Aggregate>(variant,options,count,passes,round,position);
        case 1:return run_variant<Method::GpMaterialized,Form::Aggregate>(variant,options,count,passes,round,position);
        case 2:return run_variant<Method::GpRecomputed,Form::Aggregate>(variant,options,count,passes,round,position);
        case 3:return run_variant<Method::Blocked2,Form::Aggregate>(variant,options,count,passes,round,position);
        case 4:return run_variant<Method::Original,Form::ScalarLocal>(variant,options,count,passes,round,position);
        case 5:return run_variant<Method::GpMaterialized,Form::ScalarLocal>(variant,options,count,passes,round,position);
        case 6:return run_variant<Method::GpRecomputed,Form::ScalarLocal>(variant,options,count,passes,round,position);
        case 7:return run_variant<Method::Blocked2,Form::ScalarLocal>(variant,options,count,passes,round,position);
    }
    throw std::logic_error("unknown variant dispatch");
}
template<Method M,Form F>
void validate_prefix(Word seed,Word count) {
    Input reference=make_input(seed),candidate=reference;
    for (Word i=0;i<count;++i) {
        pa_l0_state::aggregate_step<Method::Original>(reference);
        pa_l0_state::advance<M,F>(candidate,1,1);
        require(same_state(reference,candidate),"pre-timing full-state prefix mismatch");
    }
}
void validate_all(Word seed,Word count) {
    validate_prefix<Method::Original,Form::Aggregate>(seed,count);
    validate_prefix<Method::GpMaterialized,Form::Aggregate>(seed,count);
    validate_prefix<Method::GpRecomputed,Form::Aggregate>(seed,count);
    validate_prefix<Method::Blocked2,Form::Aggregate>(seed,count);
    validate_prefix<Method::Original,Form::ScalarLocal>(seed,count);
    validate_prefix<Method::GpMaterialized,Form::ScalarLocal>(seed,count);
    validate_prefix<Method::GpRecomputed,Form::ScalarLocal>(seed,count);
    validate_prefix<Method::Blocked2,Form::ScalarLocal>(seed,count);
}
std::array<unsigned,8> initial_order(Word seed) {
    seed^=UINT64_C(0xd1b54a32d192ed03);
    std::array<unsigned,8> order{{0,1,2,3,4,5,6,7}};
    for (std::size_t i=order.size()-1;i>0;--i)
        std::swap(order[i],order[splitmix64(seed)%(i+1)]);
    return order;
}
struct Collected {
    Word count,passes,prefix;
    Input reference;
    std::array<unsigned,8> order;
    std::vector<Sample> calibration,warmups,samples;
};
Collected collect(const Options& options) {
    Collected result{};
    result.count=options.smoke?16:4096;
    result.passes=options.smoke?2:1;
    result.prefix=options.smoke?64:4096;
    result.order=initial_order(options.seed);
    validate_all(options.seed,result.prefix);
    if (!options.smoke) {
        const Word target=options.target_ms*1000000;
        const Word cap=std::min(pa_l0_state::max_passes,pa_l0_state::max_operations/result.count);
        bool reached=false;
        for (unsigned trial=0;trial<8;++trial) {
            const Sample sample=dispatch(0,options,result.count,result.passes,trial,0);
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
    result.reference=make_input(options.seed);
    pa_l0_state::advance<Method::Original,Form::Aggregate>(result.reference,
                                                         std::min<Word>(result.count,4096),1);
    // Full untimed Original replay; final states are compared exactly, not just
    // by checksum. Independent Boost arithmetic gate is a separate executable.
    pa_l0_state::advance<Method::Original,Form::Aggregate>(result.reference,result.count,result.passes);
    if (!result.calibration.empty())
        require(same_state(result.calibration.back().final_state,result.reference),
                "calibration/replay complete-state mismatch");
    for (unsigned phase=0;phase<2;++phase) {
        const unsigned rounds=phase==0?(options.smoke?1:2):8;
        for (unsigned round=0;round<rounds;++round) {
            for (unsigned position=0;position<variants.size();++position) {
                const unsigned variant=result.order[(position+round)%variants.size()];
                Sample sample=dispatch(variant,options,result.count,result.passes,round,position);
                require(same_state(sample.final_state,result.reference),"sample full-state mismatch");
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
    const auto order=initial_order(20260905);
    std::array<std::array<unsigned,8>,8> appearances{};
    for (unsigned round=0;round<8;++round)
        for (unsigned position=0;position<8;++position) ++appearances[order[(position+round)%8]][position];
    for (const auto& row:appearances) for (unsigned n:row) require(n==1,"unbalanced rotation");
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
        const auto& s=samples[i]; const auto v=variants[s.variant];
        if(i)std::cout<<',';
        std::cout<<"{\"variant\":"<<s.variant<<",\"method\":"<<quote(method_name(v.method))
                 <<",\"form\":"<<quote(form_name(v.form))<<",\"round\":"<<s.round<<",\"position\":"<<s.position
                 <<",\"count\":"<<s.count<<",\"passes\":"<<s.passes<<",\"operations\":"<<s.count*s.passes
                 <<",\"start_steady_ns\":"<<s.clock.start_ns<<",\"stop_steady_ns\":"<<s.clock.stop_ns
                 <<",\"elapsed_ns\":"<<s.clock.elapsed_ns<<",\"ns_per_addition\":"<<ns_per_operation(s)
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
    std::cout<<std::setprecision(17)<<"{\"protocol\":\"parseatlas_l0_native_state_compare_v1\",\"mode\":"
             <<quote(options.smoke?"smoke_not_performance_evidence":"measurement")
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
             <<",\"state_bytes\":"<<sizeof(Input)<<",\"requested_target_ns\":"<<target
             <<",\"count\":"<<result.count<<",\"passes\":"<<result.passes
             <<",\"operations_per_sample\":"<<result.count*result.passes
             <<",\"warmup_steps_per_sample\":"<<std::min<Word>(result.count,4096)
             <<",\"warmup_rounds\":"<<(options.smoke?1:2)<<",\"sample_rounds\":8"
               ",\"measurement_workers\":1,\"statistics_parser_order_selftest\":\"passed\""
             <<",\"preflight_complete_state_checks\":"<<result.prefix*variants.size()
             <<",\"every_warmup_and_measured_final_state_compared_exactly\":true,\"reference_checksum\":"
             <<quote(hex(checksum(result.reference)))<<",\"initial_variant_order\":[";
    for (std::size_t i=0;i<result.order.size();++i) { if(i)std::cout<<',';std::cout<<result.order[i]; }
    std::cout<<"],\"conditions\":["
      "\"Four-limb addition and rotate/XOR recurrence, not field/scalar modular arithmetic\","
      "\"Four methods crossed with two source-state forms; same full boundary and recurrence\","
      "\"Unpack and full final writeback inside timer for all eight wrappers\","
      "\"Source scalarization is not a promise of register allocation; inspect actual measured binary\","
      "\"Original aggregate calibrates common passes; below-target samples are retained\","
      "\"Eight seeded rotating measured rounds cover every position once for each variant\","
      "\"Every sample reset identically; equal length same-variant warmup outside timer\","
      "\"Complete final state checked against untimed Original replay; separate Boost gate is independent\","
      "\"All calibration, warmup and measured samples retained; ratios descriptive, not confidence intervals\","
      "\"One pinned serial measurement worker prevents benchmark self-interference\","
      "\"Only own-process affinity changes; no sudo, governor, turbo, sysctl or other process changes\","
      "\"External load, sibling activity, frequency and thermal state are uncontrolled\","
      "\"No Python, PMU counters, novelty, constant-time proof, production accept or engine-wide claim\"]"
      ",\"calibration\":";
    print_samples(result.calibration,target);std::cout<<",\"warmups\":";print_samples(result.warmups,target);
    std::cout<<",\"samples\":";print_samples(result.samples,target);std::cout<<",\"summaries\":[";
    for (unsigned variant=0;variant<variants.size();++variant) {
        std::vector<double> elapsed,vs_aggregate,vs_form;
        unsigned below=0;
        for (const auto& sample:result.samples) if(sample.variant==variant) {
            elapsed.push_back(ns_per_operation(sample));
            const unsigned form_original=variants[variant].form==Form::Aggregate?0:4;
            const auto paired=[&](unsigned baseline) {
                const auto found=std::find_if(result.samples.begin(),result.samples.end(),
                    [&](const Sample& s){return s.variant==baseline && s.round==sample.round;});
                require(found!=result.samples.end(),"paired baseline missing");
                return ns_per_operation(*found)/ns_per_operation(sample);
            };
            vs_aggregate.push_back(paired(0));vs_form.push_back(paired(form_original));
            if(target && sample.clock.elapsed_ns<target)++below;
        }
        if(variant)std::cout<<',';
        std::cout<<"{\"variant\":"<<variant<<",\"method\":"<<quote(method_name(variants[variant].method))
                 <<",\"form\":"<<quote(form_name(variants[variant].form))
                 <<",\"sample_count\":"<<elapsed.size()<<",\"below_target_count\":"<<below
                 <<",\"ns_per_addition\":";print_statistics(summarize(elapsed));
        std::cout<<",\"paired_original_aggregate_over_variant\":";print_statistics(summarize(vs_aggregate));
        std::cout<<",\"paired_within_form_original_over_variant\":";print_statistics(summarize(vs_form));
        std::cout<<",\"aggregate_ratio_by_round\":";print_values(vs_aggregate);
        std::cout<<",\"within_form_ratio_by_round\":";print_values(vs_form);std::cout<<'}';
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
        std::cerr<<"l0_state_compare: "<<error.what()<<'\n';return 2;
    }
}
