#include <klee/klee.h>
#include <cstdint>
#include <cstring>
#include <cstddef>

static int der_parse_model(const uint8_t* der, size_t len) {
    if (len < 8 || len > 72) return -1;
    if (der[0] != 0x30) return -2;
    size_t total = der[1];
    if (total + 2 != len) return -3;
    if (der[2] != 0x02) return -4;
    size_t r_len = der[3];
    if (r_len == 0 || r_len > 33 || 4 + r_len >= len) return -5;
    if (der[4 + r_len] != 0x02) return -6;
    size_t s_len = der[5 + r_len];
    if (s_len == 0 || s_len > 33) return -7;
    if (5 + r_len + s_len != len) return -8;
    if (r_len > 1 && der[4] == 0x00 && (der[5] & 0x80) == 0) return -9;
    if (s_len > 1 && der[5 + r_len] == 0x00 && (der[6 + r_len] & 0x80) == 0) return -10;
    return 0;
}

#define SIG_LEN 72

int main() {
    uint8_t sig[SIG_LEN];
    klee_make_symbolic(sig, sizeof(sig), "der_sig");
    size_t len;
    klee_make_symbolic(&len, sizeof(len), "sig_len");
    klee_assume(len >= 8 && len <= SIG_LEN);
    der_parse_model(sig, len);
    return 0;
}
