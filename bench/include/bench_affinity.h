// bench_affinity.h -- CPU affinity helpers (Windows + Linux)
#pragma once

#include <cstdio>

#ifdef _WIN32
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
#  include <unistd.h>
#endif

namespace bench {

inline bool pin_to_core(int core_id) {
    if (core_id < 0) return false;
#ifdef _WIN32
    DWORD_PTR mask = static_cast<DWORD_PTR>(1) << core_id;
    HANDLE thread = GetCurrentThread();
    if (SetThreadAffinityMask(thread, mask) == 0) {
        std::fprintf(stderr, "[bench] WARNING: SetThreadAffinityMask(%d) failed\n", core_id);
        return false;
    }
    // also set high priority
    SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
    return true;
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        std::fprintf(stderr, "[bench] WARNING: sched_setaffinity(%d) failed\n", core_id);
        return false;
    }
    return true;
#else
    (void)core_id;
    std::fprintf(stderr, "[bench] WARNING: CPU affinity not supported on this platform\n");
    return false;
#endif
}

inline void print_cpu_info() {
    std::printf("--- CPU Info ---\n");
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    std::printf("  Logical cores: %u\n", static_cast<unsigned>(si.dwNumberOfProcessors));
    std::printf("  Arch: %u\n", static_cast<unsigned>(si.wProcessorArchitecture));
#elif defined(__linux__)
    // try /proc/cpuinfo model name
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                std::printf("  %s", line);
                break;
            }
        }
        fclose(f);
    }
    std::printf("  Logical cores: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

} // namespace bench
