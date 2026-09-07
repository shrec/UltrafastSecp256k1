// F1 final-only reduction benchmark; native C++, research-only, no production edits.
#include "f1_reduce.hpp"
#include "secp256k1/scalar.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {
using pa_f1::Word;
using pa_f1::Limbs;
using Scalar=secp256k1::fast::Scalar;
using Clock=std::chrono::steady_clock;
enum class Method { Serial=0, Chunk4=1, Tree=2, ColdMulti=3 };
constexpr std::array<const char*,4> names{{
    "serial", "chunk4_contiguous_interleaved", "pairwise_tree_full_allocation", "cold_multicore"}};
constexpr Word max_jobs=1000000, max_inputs_per_region=1000000000;
constexpr Word max_thread_launches_per_region=1000000;

void require(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message); // Live with NDEBUG.
}
Word decimal(const std::string& input) {
    if (input.empty()) throw std::invalid_argument("empty unsigned decimal");
    Word result=0;
    for (char ch:input) {
        if (ch<'0'||ch>'9') throw std::invalid_argument("invalid unsigned decimal");
        const Word digit=static_cast<Word>(ch-'0');
        if (result>(std::numeric_limits<Word>::max()-digit)/10)
            throw std::invalid_argument("unsigned decimal overflow");
        result=result*10+digit;
    }
    return result;
}
bool valid_count(Word count) { return count==256 || count==4096 || count==65536 || count==1048576; }
struct Options {
    bool smoke=false;
    unsigned cpu=4;
    Word count=256,seed=20260905,target_ms=200;
};
Options options_from(int argc,char** argv) {
    if (argc<2 || (std::string(argv[1])!="--smoke" && std::string(argv[1])!="--measure"))
        throw std::invalid_argument("usage: f1_reduce_compare (--smoke|--measure) "
            "[--count 256|4096|65536|1048576] [--cpu N] [--seed N] [--target-ms 200..1000]");
    Options result;
    result.smoke=std::string(argv[1])=="--smoke";
    bool count=false,cpu=false,seed=false,target=false;
    for (int i=2;i<argc;i+=2) {
        if (i+1>=argc) throw std::invalid_argument("missing option value");
        const std::string key=argv[i];
        const Word value=decimal(argv[i+1]);
        if (key=="--count"&&!count) {
            if (!valid_count(value)) throw std::invalid_argument("unsupported count");
            result.count=value;count=true;
        } else if (key=="--cpu"&&!cpu) {
            if (value>=CPU_SETSIZE) throw std::invalid_argument("CPU exceeds CPU_SETSIZE");
            result.cpu=static_cast<unsigned>(value);cpu=true;
        } else if (key=="--seed"&&!seed) { result.seed=value;seed=true;
        } else if (key=="--target-ms"&&!target) {
            if (value<200 || value>1000) throw std::invalid_argument("target must be 200..1000 ms");
            result.target_ms=value;target=true;
        } else throw std::invalid_argument("unknown or duplicate option: "+key);
    }
    if (result.smoke&&target) throw std::invalid_argument("smoke has no performance target");
    return result;
}
std::string read_line(const std::string& path) {
    std::ifstream file(path);
    std::string value;
    if (!std::getline(file,value)) return {};
    while (!value.empty() && (value.back()=='\r'||value.back()==' '||value.back()=='\t')) value.pop_back();
    return value;
}
bool read_nonnegative(const std::string& path,Word& output) {
    try {
        output=decimal(read_line(path));
        return true;
    } catch (const std::invalid_argument&) { return false; }
}
struct CoreRecord { int cpu; Word package,core; };
struct Topology {
    std::vector<int> allowed,workers,reserved;
    std::vector<CoreRecord> records;
    bool physical=false;
    Word observed_units=0;
    long online_logical=0;
};
// One representative per observable physical core when topology is complete.
// Reserve the coordinator's entire core and one more unit when possible.
// Fallback uses unique allowed logical CPUs and explicitly does NOT claim
// physical-core separation. The worker count is never hardcoded.
Topology select_workers(unsigned coordinator) {
    cpu_set_t mask;CPU_ZERO(&mask);
    require(sched_getaffinity(0,sizeof(mask),&mask)==0,"sched_getaffinity failed");
    Topology t;
    t.online_logical=sysconf(_SC_NPROCESSORS_ONLN);
    for (int cpu=0;cpu<CPU_SETSIZE;++cpu) if (CPU_ISSET(cpu,&mask)) t.allowed.push_back(cpu);
    require(std::find(t.allowed.begin(),t.allowed.end(),static_cast<int>(coordinator))!=t.allowed.end(),
            "coordinator outside inherited affinity");
    t.physical=true;
    for (int cpu:t.allowed) {
        const std::string base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
        Word package=0,core=0;
        if (!read_nonnegative(base+"physical_package_id",package) ||
            !read_nonnegative(base+"core_id",core)) { t.physical=false;break; }
        t.records.push_back({cpu,package,core});
    }
    if (t.physical) {
        std::map<std::pair<Word,Word>,int> representatives;
        std::pair<Word,Word> coordinator_core{};
        for (const auto& record:t.records) {
            representatives.emplace(std::make_pair(record.package,record.core),record.cpu);
            if (record.cpu==static_cast<int>(coordinator))
                coordinator_core={record.package,record.core};
        }
        t.observed_units=representatives.size();
        for (const auto& entry:representatives) {
            if (entry.first==coordinator_core) t.reserved.push_back(entry.second);
            else t.workers.push_back(entry.second);
        }
    } else {
        t.observed_units=t.allowed.size();
        for (int cpu:t.allowed)
            (cpu==static_cast<int>(coordinator)?t.reserved:t.workers).push_back(cpu);
    }
    if (t.workers.size()>1) {
        t.reserved.push_back(t.workers.back());
        t.workers.pop_back();
    }
    // A one-unit environment cannot provide reserved-worker separation.
    // Run one explicitly labeled colocated worker instead of inventing cores.
    if (t.workers.empty()) t.workers.push_back(static_cast<int>(coordinator));
    return t;
}
void pin_coordinator(unsigned cpu) {
    cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);
    require(sched_setaffinity(0,sizeof(mask),&mask)==0,"coordinator affinity failed");
    require(sched_getcpu()==static_cast<int>(cpu),"coordinator CPU not established");
}
struct CpuEndpoint { int cpu;std::string frequency,governor; };
struct EnvironmentEndpoint { std::vector<CpuEndpoint> cpus;std::string no_turbo; };
EnvironmentEndpoint environment_endpoint(const Options& options,const Topology& topology) {
    EnvironmentEndpoint result;
    std::vector<int> cpus=topology.workers;
    cpus.push_back(static_cast<int>(options.cpu));
    std::sort(cpus.begin(),cpus.end());cpus.erase(std::unique(cpus.begin(),cpus.end()),cpus.end());
    for (int cpu:cpus) {
        const std::string base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/cpufreq/";
        result.cpus.push_back({cpu,read_line(base+"scaling_cur_freq"),read_line(base+"scaling_governor")});
    }
    result.no_turbo=read_line("/sys/devices/system/cpu/intel_pstate/no_turbo");
    return result;
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
        if (pa_modn::canonical(value)) return value;
    }
    throw std::runtime_error("canonical rejection cap reached");
}
struct Inputs { Limbs x0;std::vector<Limbs> rhs; };
Inputs make_inputs(const Options& options) {
    require(valid_count(options.count) && options.count<=pa_f1::max_count,"invalid input extent");
    Word seed=options.seed;
    Inputs result{random_canonical(seed),{}};
    result.rhs.reserve(static_cast<std::size_t>(options.count));
    for (Word i=0;i<options.count;++i) result.rhs.push_back(random_canonical(seed));
    return result;
}
bool exact(const Limbs& a,const Limbs& b) { return pa_modn::encode_le(a)==pa_modn::encode_le(b); }
Word checksum(const Limbs& value) {
    Word result=UINT64_C(0xcbf29ce484222325);
    for (auto byte:pa_modn::encode_le(value)) result=(result^byte)*UINT64_C(0x100000001b3);
    return result;
}
Word input_checksum(const Inputs& input) {
    Word result=checksum(input.x0);
    for (const auto& value:input.rhs)
        for (auto byte:pa_modn::encode_le(value)) result=(result^byte)*UINT64_C(0x100000001b3);
    return result;
}
inline void observe(const void* pointer) { asm volatile("" : : "g"(pointer) : "memory"); }
#if defined(__GNUC__) && !defined(__clang__)
#define PA_F1_JOB_BOUNDARY __attribute__((noinline,noipa))
#else
#define PA_F1_JOB_BOUNDARY __attribute__((noinline))
#endif
// Common non-IPA job boundary preserves repeated FULL jobs, not periodic-stream
// summarization. Memory clobbers are per complete job, never per modular add.
// Output materialization is intentionally part of the same boundary for all.
template<Method M>
PA_F1_JOB_BOUNDARY Limbs full_job(const Inputs& inputs,const std::vector<int>& cpus) {
    observe(&inputs.x0);observe(inputs.rhs.data());
    Limbs result{};
    if constexpr (M==Method::Serial)
        result=pa_f1::serial(inputs.x0,inputs.rhs.data(),inputs.rhs.size());
    else if constexpr (M==Method::Chunk4)
        result=pa_f1::chunk4_interleaved(inputs.x0,inputs.rhs.data(),inputs.rhs.size());
    else if constexpr (M==Method::Tree) {
        // No reused workspace: allocate, initialize, copy, reduce and free on
        // every invocation. A persistent allocator may still recycle storage.
        std::vector<Limbs> scratch(inputs.rhs.size());
        result=pa_f1::tree(inputs.x0,inputs.rhs.data(),inputs.rhs.size(),scratch.data(),scratch.size());
    } else result=pa_f1::multicore_cold(inputs.x0,inputs.rhs.data(),inputs.rhs.size(),cpus);
    observe(&result);
    return result;
}
Word job_cap(unsigned variant,Word count,Word workers) {
    if (variant>=4 || !valid_count(count) || !workers) return 0;
    Word cap=std::min(max_jobs,max_inputs_per_region/count);
    if (variant==3) cap=std::min(cap,max_thread_launches_per_region/workers);
    return cap;
}
bool valid_work(unsigned variant,Word count,Word jobs,Word workers) {
    return jobs>0 && jobs<=job_cap(variant,count,workers);
}
struct RegionClock { Word start_ns,stop_ns,elapsed_ns; };
struct Sample {
    unsigned variant,round,position;
    Word count,jobs;
    RegionClock clock;
    Limbs final;
    int cpu_before,cpu_after;
};
template<Method M>
__attribute__((noinline)) RegionClock timed_region(const Inputs& inputs,
    const std::vector<int>& cpus,Word jobs,Limbs& final) {
    observe(&inputs);observe(&final);
    const auto start=Clock::now();
    for (Word job=0;job<jobs;++job) final=full_job<M>(inputs,cpus);
    observe(&final);
    const auto stop=Clock::now();
    const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
    require(elapsed>0,"nonpositive elapsed time");
    return {static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
               start.time_since_epoch()).count()),
            static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop.time_since_epoch()).count()),static_cast<Word>(elapsed)};
}
template<Method M>
Sample run(const Options& options,const Inputs& inputs,const Topology& topology,Word jobs,
           unsigned round,unsigned position) {
    constexpr unsigned variant=static_cast<unsigned>(M);
    const Word workers=std::min<Word>(options.count,topology.workers.size());
    require(valid_work(variant,options.count,jobs,workers),"invalid count/jobs/resource bounds");
    const int before=sched_getcpu();
    require(before==static_cast<int>(options.cpu),"coordinator CPU changed before sample");
    Limbs final{};
    const auto clock=timed_region<M>(inputs,topology.workers,jobs,final);
    const int after=sched_getcpu();
    require(after==static_cast<int>(options.cpu),"coordinator CPU changed after sample");
    return {variant,round,position,options.count,jobs,clock,final,before,after};
}
Sample dispatch(unsigned variant,const Options& options,const Inputs& inputs,const Topology& topology,
                Word jobs,unsigned round,unsigned position) {
    switch (variant) {
        case 0:return run<Method::Serial>(options,inputs,topology,jobs,round,position);
        case 1:return run<Method::Chunk4>(options,inputs,topology,jobs,round,position);
        case 2:return run<Method::Tree>(options,inputs,topology,jobs,round,position);
        case 3:return run<Method::ColdMulti>(options,inputs,topology,jobs,round,position);
    }
    throw std::logic_error("unknown method");
}
std::array<unsigned,4> initial_order(Word seed) {
    seed^=UINT64_C(0xd1b54a32d192ed03);
    std::array<unsigned,4> result{{0,1,2,3}};
    for (std::size_t i=3;i>0;--i) std::swap(result[i],result[splitmix64(seed)%(i+1)]);
    return result;
}
struct Collected {
    std::array<Word,4> jobs{};
    std::array<unsigned,4> order;
    Limbs reference;
    Word input_digest;
    std::vector<Sample> preflight,calibration,warmups,samples;
};
Collected collect(const Options& options,const Topology& topology) {
    Collected result{};
    result.order=initial_order(options.seed);
    const Inputs inputs=make_inputs(options);
    result.input_digest=input_checksum(inputs);
    Scalar reference=Scalar::from_limbs(inputs.x0);
    for (const auto& value:inputs.rhs) reference+=Scalar::from_limbs(value);
    result.reference=reference.limbs();
    require(pa_modn::canonical(result.reference),"noncanonical API reference");
    const auto check=[&](const Sample& sample) {
        require(exact(sample.final,result.reference),"full final bytes disagree with public Scalar replay");
    };
    for (unsigned position=0;position<4;++position) {
        const unsigned variant=result.order[position];
        auto sample=dispatch(variant,options,inputs,topology,1,0,position);
        check(sample);result.preflight.push_back(sample);
    }
    for (unsigned position=0;position<4;++position) {
        const unsigned variant=result.order[position];
        result.jobs[variant]=options.smoke?2:1;
        if (options.smoke) continue;
        const Word cap=job_cap(variant,options.count,std::min<Word>(options.count,topology.workers.size()));
        bool reached=false;
        for (unsigned trial=0;trial<8;++trial) {
            const auto sample=dispatch(variant,options,inputs,topology,result.jobs[variant],trial,position);
            check(sample);result.calibration.push_back(sample);
            const Word target=options.target_ms*1000000;
            if (sample.clock.elapsed_ns>=target) { reached=true;break; }
            const long double proposed=std::ceil(static_cast<long double>(result.jobs[variant])*
                                                target*1.2L/sample.clock.elapsed_ns);
            const Word next=proposed>=cap?cap:static_cast<Word>(proposed);
            require(next>result.jobs[variant],"calibration impossible within resource caps");
            result.jobs[variant]=next;
        }
        require(reached,"calibration target not reached in eight trials");
    }
    // One serial coordinator avoids concurrent benchmark-region interference.
    // Only ColdMulti's within-job transformation uses worker threads.
    for (unsigned phase=0;phase<2;++phase)
        for (unsigned round=0;round<(phase==0?2U:8U);++round)
            for (unsigned position=0;position<4;++position) {
                const unsigned variant=result.order[(position+round)%4];
                auto sample=dispatch(variant,options,inputs,topology,result.jobs[variant],round,position);
                check(sample);
                (phase==0?result.warmups:result.samples).push_back(sample);
            }
    require(input_checksum(inputs)==result.input_digest,"input content changed during invocation");
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
Statistics statistics(std::vector<double> values) {
    require(!values.empty(),"empty summary");
    for (double value:values) require(std::isfinite(value),"nonfinite summary");
    std::sort(values.begin(),values.end());
    const double median=quantile(values,.5);
    std::vector<double> deviations;
    for (double value:values) deviations.push_back(std::abs(value-median));
    std::sort(deviations.begin(),deviations.end());
    return {median,values.front(),values.back(),quantile(deviations,.5),
            quantile(values,.75)-quantile(values,.25)};
}
void selftest() {
    const auto odd=statistics({7,1,4,3,6,2,5}),even=statistics({8,1,7,2,6,3,5,4}),one=statistics({3});
    require(odd.median==4&&odd.minimum==1&&odd.maximum==7&&odd.mad==2&&odd.iqr==3,"odd statistics");
    require(even.median==4.5&&even.mad==2&&even.iqr==3.5&&one.mad==0&&one.median==3,"even/singleton statistics");
    require(decimal("18446744073709551615")==UINT64_MAX&&decimal("0")==0,"decimal endpoints");
    for (const char* bad:{"","-1","+1","1x"," 1","18446744073709551616"}) {
        bool rejected=false;
        try { static_cast<void>(decimal(bad)); } catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid decimal accepted");
    }
    require(valid_work(0,256,1,1)&&valid_work(3,1048576,1,8),"valid work rejected");
    require(!valid_work(0,0,1,1)&&!valid_work(0,256,0,1)&&
            !valid_work(0,256,max_jobs+1,1)&&!valid_work(3,256,125001,8)&&
            !valid_work(0,UINT64_MAX,UINT64_MAX,1)&&!valid_work(0,256,1,0),"invalid work accepted");
    Limbs original{};
    for (unsigned bit=0;bit<256;++bit) {
        Limbs changed=original;changed[bit/64]^=Word{1}<<(bit%64);
        require(!exact(original,changed),"byte comparator missed corruption");
    }
    require(exact(original,original),"byte comparator identity");
    const auto order=initial_order(20260905);
    std::array<std::array<unsigned,4>,4> appearances{};
    for (unsigned round=0;round<8;++round)
        for (unsigned position=0;position<4;++position) ++appearances[order[(position+round)%4]][position];
    for (const auto& row:appearances) for (unsigned n:row) require(n==2,"unbalanced order");
}
std::string quote(const std::string& text) {
    std::ostringstream out;out<<'"';
    for (unsigned char ch:text) {
        if (ch=='"'||ch=='\\') out<<'\\'<<ch;
        else if (ch<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(ch)<<std::dec;
        else out<<ch;
    }
    out<<'"';return out.str();
}
std::string hex(Word value) {
    std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<value;return out.str();
}
void print_ints(const std::vector<int>& values) {
    std::cout<<'[';
    for (std::size_t i=0;i<values.size();++i) { if(i)std::cout<<',';std::cout<<values[i]; }
    std::cout<<']';
}
void print_endpoint(const EnvironmentEndpoint& endpoint) {
    std::cout<<"{\"cpus\":[";
    for (std::size_t i=0;i<endpoint.cpus.size();++i) {
        const auto& cpu=endpoint.cpus[i];if(i)std::cout<<',';
        std::cout<<"{\"cpu\":"<<cpu.cpu<<",\"scaling_cur_freq_khz\":"
                 <<(cpu.frequency.empty()?"null":quote(cpu.frequency))
                 <<",\"scaling_governor\":"<<(cpu.governor.empty()?"null":quote(cpu.governor))<<'}';
    }
    std::cout<<"],\"intel_pstate_no_turbo\":"
             <<(endpoint.no_turbo.empty()?"null":quote(endpoint.no_turbo))<<'}';
}
double ns_per_job(const Sample& sample) { return static_cast<double>(sample.clock.elapsed_ns)/sample.jobs; }
Word additions_per_job(unsigned variant,Word count,Word workers) {
    return count+(variant==1?4:(variant==3?workers:0));
}
void print_samples(const std::vector<Sample>& samples,Word target,Word workers) {
    std::cout<<'[';
    for (std::size_t i=0;i<samples.size();++i) {
        const auto& s=samples[i];if(i)std::cout<<',';
        std::cout<<"{\"variant\":"<<s.variant<<",\"method\":"<<quote(names[s.variant])
          <<",\"round\":"<<s.round<<",\"position\":"<<s.position<<",\"count\":"<<s.count
          <<",\"jobs\":"<<s.jobs<<",\"total_rhs_inputs\":"<<s.jobs*s.count
          <<",\"source_modular_additions\":"<<s.jobs*additions_per_job(s.variant,s.count,workers)
          <<",\"source_tree_copied_scalar_elements\":"<<(s.variant==2?s.jobs*s.count:0)
          <<",\"source_tree_zero_initialized_scalar_elements\":"<<(s.variant==2?s.jobs*s.count:0)
          <<",\"tree_scratch_bytes_per_job\":"<<(s.variant==2?s.count*32:0)
          <<",\"tree_vector_storage_requests\":"<<(s.variant==2?s.jobs:0)
          <<",\"cold_partial_bytes_per_job\":"<<(s.variant==3?workers*32:0)
          <<",\"cold_vector_storage_requests\":"<<(s.variant==3?s.jobs*3:0)
          <<",\"source_output_bytes_materialized\":"<<s.jobs*32
          <<",\"thread_launches\":"<<(s.variant==3?s.jobs*workers:0)
          <<",\"start_steady_ns\":"<<s.clock.start_ns<<",\"stop_steady_ns\":"<<s.clock.stop_ns
          <<",\"elapsed_ns\":"<<s.clock.elapsed_ns<<",\"region_mean_ns_per_full_job\":"<<ns_per_job(s)
          <<",\"region_mean_ns_per_rhs_input\":"<<ns_per_job(s)/s.count
          <<",\"final_checksum\":"<<quote(hex(checksum(s.final)))
          <<",\"cpu_before\":"<<s.cpu_before<<",\"cpu_after\":"<<s.cpu_after
          <<",\"below_requested_target\":";
        if(target)std::cout<<(s.clock.elapsed_ns<target?"true":"false");else std::cout<<"null";
        std::cout<<'}';
    }
    std::cout<<']';
}
void print_stats(const Statistics& value) {
    std::cout<<"{\"median\":"<<value.median<<",\"min\":"<<value.minimum<<",\"max\":"<<value.maximum
             <<",\"mad\":"<<value.mad<<",\"inclusive_iqr\":"<<value.iqr<<'}';
}
Word unix_now_ns() {
    return static_cast<Word>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
void report(const Options& options,const Topology& topology,const Collected& result,
            const EnvironmentEndpoint& before,const EnvironmentEndpoint& after,Word started,Word finished) {
    const Word target=options.smoke?0:options.target_ms*1000000;
    const Word workers=std::min<Word>(options.count,topology.workers.size());
    std::cout<<std::setprecision(17)<<"{\"protocol\":\"parseatlas_f1_final_reduction_v1\",\"mode\":"
      <<quote(options.smoke?"smoke_not_performance_evidence":"measurement")
      <<",\"timing_claim\":"<<(options.smoke?"false":"true")
      <<",\"started_unix_ns\":"<<started<<",\"finished_unix_ns\":"<<finished
      <<",\"compiler_version\":"<<quote(__VERSION__)<<",\"cplusplus\":"<<__cplusplus
      <<",\"count\":"<<options.count<<",\"rhs_bytes\":"<<options.count*32
      <<",\"seed\":"<<options.seed<<",\"coordinator_cpu\":"<<options.cpu
      <<",\"requested_target_ns\":"<<target<<",\"topology\":{\"inherited_allowed_cpus\":";
    print_ints(topology.allowed);
    std::cout<<",\"online_logical_sysconf\":"<<topology.online_logical
      <<",\"selection_unit\":"<<quote(topology.physical?"physical_core":"logical_cpu_fallback")
      <<",\"observed_allowed_units\":"<<topology.observed_units<<",\"worker_cpus\":";
    print_ints(topology.workers);std::cout<<",\"reserved_unit_representatives\":";print_ints(topology.reserved);
    std::cout<<",\"worker_count\":"<<workers<<",\"coordinator_colocated_worker\":"
      <<(std::find(topology.workers.begin(),topology.workers.end(),static_cast<int>(options.cpu))
         !=topology.workers.end()?"true":"false")<<",\"topology_records\":[";
    for (std::size_t i=0;i<topology.records.size();++i) {
        if(i)std::cout<<',';
        const auto& r=topology.records[i];
        std::cout<<"{\"cpu\":"<<r.cpu<<",\"package\":"<<r.package<<",\"core\":"<<r.core<<'}';
    }
    std::cout<<"]},\"environment_before\":";print_endpoint(before);
    std::cout<<",\"environment_after\":";print_endpoint(after);
#ifdef SECP256K1_NO_INT128
    constexpr bool no_int128=true;
#else
    constexpr bool no_int128=false;
#endif
#ifdef __SIZEOF_INT128__
    constexpr bool has_int128=true;
#else
    constexpr bool has_int128=false;
#endif
    std::cout<<",\"original_backend\":"<<quote(no_int128||!has_int128?"portable_carry":"gcc_uint128")
      <<",\"no_int128_defined\":"<<(no_int128?"true":"false")
      <<",\"int128_macro_defined\":"<<(has_int128?"true":"false")
      <<",\"noipa_available\":"
#if defined(__GNUC__) && !defined(__clang__)
      <<"true"
#else
      <<"false"
#endif
      <<",\"input_checksum_before_and_after\":"<<quote(hex(result.input_digest))
      <<",\"reference_final_checksum\":"<<quote(hex(checksum(result.reference)))
      <<",\"comparison\":\"exact_32_little_endian_bytes_against_public_Scalar\""
        ",\"every_recorded_sample_final_result_compared_exactly\":true"
        ",\"every_individual_job_result_checked\":false"
        ",\"statistics_parser_work_order_comparator_selftest\":\"passed\""
        ",\"warmup_rounds\":2,\"measured_rounds\":8,\"jobs_by_variant\":[";
    for (unsigned v=0;v<4;++v) { if(v)std::cout<<',';std::cout<<result.jobs[v]; }
    std::cout<<"],\"initial_variant_order\":[";
    for (unsigned v=0;v<4;++v) { if(v)std::cout<<',';std::cout<<result.order[v]; }
    std::cout<<"],\"conditions\":["
      "\"One recurrence final state only; intermediate prefixes are not observable\","
      "\"All four representations reuse unchanged Original modular addition\","
      "\"Each full job consumes the entire same runtime RHS and incorporates x0 exactly once\","
      "\"No repeated-job summaries; common noinline job and compiler memory boundaries, no per-add barriers\","
      "\"Every job materializes 32 output bytes; only last identical-job output is checked per timed sample\","
      "\"Tree allocation, source zero-initialization, copy, reduction and free are inside each full job\","
      "\"Cold multicore includes allocation, thread creation, explicit affinity, merge, join and destruction\","
      "\"Cold means thread lifecycle per job, not cold caches, fresh allocator pages or fresh OS resources\","
      "\"Per-variant calibration gives different job counts and total work volumes\","
      "\"Ratios compare same-round normalized region means, not matched-work or individual-job latency\","
      "\"Source operation/copy counts are semantic accounting, not dynamic instructions or PMU memory traffic\","
      "\"No per-sample prewarm; two full warmup rounds and all calibration records retained\","
      "\"All below-target samples retained; median/MAD/IQR summarize region means\","
      "\"Topology is discovered before coordinator pinning; each worker sets its own explicit CPU affinity\","
      "\"Frequency/governor endpoints do not establish fixed in-region frequency\","
      "\"Only own process/thread affinity changes; external load and thermals remain uncontrolled\","
      "\"No Python, constant-time proof, novelty, production replacement or whole-engine claim\"]"
      ",\"preflight\":";
    print_samples(result.preflight,0,workers);
    std::cout<<",\"calibration\":";print_samples(result.calibration,target,workers);
    std::cout<<",\"warmups\":";print_samples(result.warmups,target,workers);
    std::cout<<",\"samples\":";print_samples(result.samples,target,workers);
    std::cout<<",\"summaries\":[";
    for (unsigned variant=0;variant<4;++variant) {
        std::vector<double> means,per_input,ratios;
        unsigned below=0;
        for (const auto& sample:result.samples) if(sample.variant==variant) {
            const double mean=ns_per_job(sample);
            means.push_back(mean);per_input.push_back(mean/sample.count);
            const auto baseline=std::find_if(result.samples.begin(),result.samples.end(),
                [&](const Sample& s){return s.variant==0&&s.round==sample.round;});
            require(baseline!=result.samples.end(),"same-round serial missing");
            ratios.push_back(ns_per_job(*baseline)/mean);
            if(target&&sample.clock.elapsed_ns<target)++below;
        }
        if(variant)std::cout<<',';
        std::cout<<"{\"variant\":"<<variant<<",\"method\":"<<quote(names[variant])
          <<",\"sample_count\":"<<means.size()<<",\"below_target_count\":"<<below
          <<",\"region_mean_ns_per_full_job\":";print_stats(statistics(means));
        std::cout<<",\"region_mean_ns_per_rhs_input\":";print_stats(statistics(per_input));
        std::cout<<",\"same_round_serial_over_candidate_normalized_region_mean\":";print_stats(statistics(ratios));
        std::cout<<",\"normalized_ratio_by_round\":[";
        for (std::size_t i=0;i<ratios.size();++i) { if(i)std::cout<<',';std::cout<<ratios[i]; }
        std::cout<<"]}";
    }
    std::cout<<"]}\n";
}
} // namespace

int main(int argc,char** argv) {
    try {
        const auto options=options_from(argc,argv);
        selftest();
        const auto topology=select_workers(options.cpu);
        const auto before=environment_endpoint(options,topology);
        pin_coordinator(options.cpu);
        const Word started=unix_now_ns();
        const auto result=collect(options,topology);
        const Word finished=unix_now_ns();
        const auto after=environment_endpoint(options,topology);
        report(options,topology,result,before,after,started,finished);
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"f1_reduce_compare: "<<error.what()<<'\n';return 2;
    }
}
