// ============================================================================
// shim_gpu.cpp -- GPU backend init/shutdown for the libsecp256k1 shim
// ============================================================================
// Reads the [gpu] section from config.ini and initialises the best available
// GPU backend for non-CT verification acceleration.
//
// Config.ini [gpu] section:
//
//   [gpu]
//   enabled = true          ; master switch (default: false)
//   backend = auto          ; auto | cuda | opencl | metal (default: auto)
//
// Backend probe order when backend=auto:
//   1. CUDA   (UFSECP_GPU_BACKEND_CUDA   = 1)  — NVIDIA discrete + Jetson
//   2. OpenCL (UFSECP_GPU_BACKEND_OPENCL = 2)  — Intel iGPU / AMD / any CL
//   3. Metal  (UFSECP_GPU_BACKEND_METAL  = 3)  — Apple Silicon
//
// CT signing NEVER uses the GPU path. Only non-CT operations (ECDSA verify,
// Schnorr verify) are dispatched to GPU when available.
// ============================================================================

#ifdef SECP256K1_SHIM_GPU

#include "shim_gpu_state.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

// ── Config parser -----------------------------------------------------------

namespace {

// Minimal ini parser: finds key=value inside [section].
// Returns empty string if not found.
static std::string ini_read(const char* path,
                             const char* section,
                             const char* key)
{
    std::FILE* f = std::fopen(path, "r");
    if (!f) return {};

    char line[512];
    bool in_section = (section == nullptr); // null section = global scope

    while (std::fgets(line, sizeof(line), f)) {
        // Strip trailing newline / CR
        size_t len = std::strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        // Skip blank lines and comments
        const char* p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '\0' || *p == '#' || *p == ';') continue;

        // Section header
        if (*p == '[') {
            const char* end = std::strchr(p + 1, ']');
            if (!end) continue;
            std::string sec(p + 1, end);
            in_section = (section && sec == section);
            continue;
        }

        if (!in_section) continue;

        // key = value
        const char* eq = std::strchr(p, '=');
        if (!eq) continue;

        std::string k(p, eq);
        // trim right of key
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();

        if (k != key) continue;

        std::string v(eq + 1);
        // trim both ends
        size_t s = v.find_first_not_of(" \t");
        size_t e = v.find_last_not_of(" \t\r\n");
        if (s == std::string::npos) { std::fclose(f); return {}; }
        v = v.substr(s, e - s + 1);
        // strip inline comment
        auto com = v.find_first_of("#;");
        if (com != std::string::npos) {
            v = v.substr(0, com);
            while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
        }
        std::fclose(f);
        return v;
    }
    std::fclose(f);
    return {};
}

static bool ini_read_bool(const char* path, const char* sec, const char* key,
                           bool default_val = false)
{
    auto v = ini_read(path, sec, key);
    if (v.empty()) return default_val;
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return (v == "true" || v == "1" || v == "yes");
}

static uint32_t ini_read_uint(const char* path, const char* sec, const char* key,
                               uint32_t default_val = 0)
{
    auto v = ini_read(path, sec, key);
    if (v.empty()) return default_val;
    try { return static_cast<uint32_t>(std::stoul(v)); }
    catch (...) { return default_val; }
}

// Probe order for auto-detect
static constexpr uint32_t kProbeOrder[] = {
    UFSECP_GPU_BACKEND_CUDA,
    UFSECP_GPU_BACKEND_OPENCL,
    UFSECP_GPU_BACKEND_METAL,
};

// Maps config platform name → ufsecp backend ID.
// Aliases:
//   cuda   → CUDA  (NVIDIA discrete, Jetson)
//   opencl → OpenCL (Intel iGPU, AMD via ROCm-OpenCL, NVIDIA via CL)
//   metal  → Metal (Apple Silicon / macOS)
//   rocm   → OpenCL (AMD ROCm OpenCL runtime — no dedicated backend ID yet)
//   hip    → CUDA  (AMD HIP via CUDA portability layer)
// Returns NONE for "auto" or unrecognised — triggers probe-order fallback.
static uint32_t name_to_backend(const std::string& name) {
    if (name == "cuda" || name == "hip")           return UFSECP_GPU_BACKEND_CUDA;
    if (name == "opencl" || name == "rocm")        return UFSECP_GPU_BACKEND_OPENCL;
    if (name == "metal")                           return UFSECP_GPU_BACKEND_METAL;
    return UFSECP_GPU_BACKEND_NONE; // "auto" or unknown → probe-order
}

} // anonymous namespace

// ── Singleton ---------------------------------------------------------------

static ShimGpuState g_gpu_state;

ShimGpuState& shim_gpu_state() noexcept { return g_gpu_state; }

// ── Init / shutdown ---------------------------------------------------------

static std::once_flag g_gpu_init_once;

void shim_gpu_init(const char* config_ini_path)
{
    std::call_once(g_gpu_init_once, [config_ini_path]() {
        auto& gs = g_gpu_state;

        // Read master switch
        const char* path = config_ini_path ? config_ini_path : "config.ini";
        if (!ini_read_bool(path, "gpu", "enabled", false)) {
            return; // GPU disabled in config — stay on CPU
        }

        // Determine backend/platform preference.
        // Accept both "platform" and "backend" keys — "platform" is the
        // user-facing term; "backend" kept for backward compatibility.
        // Valid values: auto | cuda | opencl | metal | rocm | hip
        std::string backend_str = ini_read(path, "gpu", "platform");
        if (backend_str.empty())
            backend_str = ini_read(path, "gpu", "backend");
        std::transform(backend_str.begin(), backend_str.end(),
                       backend_str.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        uint32_t preferred = name_to_backend(backend_str); // 0 = auto

        // Device index: which GPU to use within the selected backend (default: 0)
        uint32_t device_index = ini_read_uint(path, "gpu", "device", 0);

        // Build probe list
        uint32_t probe[4];
        size_t   nprobe = 0;
        if (preferred != UFSECP_GPU_BACKEND_NONE) {
            probe[nprobe++] = preferred;
        } else {
            for (auto b : kProbeOrder) probe[nprobe++] = b;
        }

        // Try each backend until one works
        for (size_t i = 0; i < nprobe; ++i) {
            uint32_t bid = probe[i];
            if (!ufsecp_gpu_is_available(bid)) continue;

            // Clamp device_index to what's actually present
            uint32_t dev_count = ufsecp_gpu_device_count(bid);
            uint32_t dev       = (device_index < dev_count) ? device_index : 0;

            ufsecp_gpu_ctx* ctx = nullptr;
            ufsecp_error_t  rc  = ufsecp_gpu_ctx_create(&ctx, bid, dev);
            if (rc != UFSECP_OK || !ctx) continue;

            gs.ctx     = ctx;
            gs.backend = bid;
            gs.device  = dev;
            gs.enabled = true;

            // Log to stderr so the user can see which backend was selected
            ufsecp_gpu_device_info_t info{};
            ufsecp_gpu_device_info(bid, dev, &info);
            std::fprintf(stderr,
                "[secp256k1-shim] GPU acceleration: %s device %u — %s (%u CUs, %u MHz)\n",
                ufsecp_gpu_backend_name(bid), dev,
                info.name, info.compute_units, info.max_clock_mhz);
            return;
        }

        // No usable backend found despite enabled=true
        std::fprintf(stderr,
            "[secp256k1-shim] GPU acceleration requested but no backend available"
            " — falling back to CPU.\n");
    });
}

void shim_gpu_shutdown()
{
    auto& gs = g_gpu_state;
    std::lock_guard<std::mutex> lk(gs.mu);
    if (gs.ctx) {
        ufsecp_gpu_ctx_destroy(gs.ctx);
        gs.ctx     = nullptr;
        gs.backend = UFSECP_GPU_BACKEND_NONE;
        gs.enabled = false;
    }
}

#endif // SECP256K1_SHIM_GPU
