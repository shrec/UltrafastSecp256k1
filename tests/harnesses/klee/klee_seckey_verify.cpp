#include <klee/klee.h>
#include <cstdint>
#include <cstring>

static const uint8_t N[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

static int seckey_verify_model(const uint8_t k[32]) {
    bool all_zero = true;
    for (int i = 0; i < 32; ++i) if (k[i]) { all_zero = false; break; }
    if (all_zero) return -1;
    for (int i = 0; i < 32; ++i) {
        if (k[i] < N[i]) return 0;
        if (k[i] > N[i]) return -2;
    }
    return -2;
}

int main() {
    uint8_t key[32];
    klee_make_symbolic(key, sizeof(key), "seckey");
    int result = seckey_verify_model(key);
    bool all_zero = true;
    for (int i = 0; i < 32; ++i) if (key[i]) { all_zero = false; break; }
    if (all_zero) { klee_assert(result != 0); }
    bool eq_n = true;
    for (int i = 0; i < 32; ++i) if (key[i] != N[i]) { eq_n = false; break; }
    if (eq_n) { klee_assert(result != 0); }
    return 0;
}
