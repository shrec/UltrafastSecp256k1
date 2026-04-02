#include <klee/klee.h>
#include <cstdint>
#include <cstring>
#include <cstddef>

static bool field_is_in_range(const uint8_t x[32]) {
    static const uint8_t P[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xFC,0x2F
    };
    for (int i = 0; i < 32; ++i) {
        if (x[i] < P[i]) return true;
        if (x[i] > P[i]) return false;
    }
    return false;
}

static int pubkey_parse_model(const uint8_t input[33]) {
    uint8_t prefix = input[0];
    if (prefix != 0x02 && prefix != 0x03) return -1;
    const uint8_t* x = input + 1;
    if (!field_is_in_range(x)) return -2;
    return 0;
}

int main() {
    uint8_t input[33];
    klee_make_symbolic(input, sizeof(input), "pubkey_bytes");
    int result = pubkey_parse_model(input);
    if (input[0] != 0x02 && input[0] != 0x03) {
        klee_assert(result == -1);
    }
    return 0;
}
