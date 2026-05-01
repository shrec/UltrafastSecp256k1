#include <stdint.h>
void fe52_mul_inner(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    unsigned __int128 c, d;
    uint64_t t3, t4, tx, u0;
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t R52 = 0x1000003D10ULL;
    const uint64_t M52 = 0xFFFFFFFFFFFFFULL;
    d  = (unsigned __int128)a0 * b[3] + (unsigned __int128)a1 * b[2] + (unsigned __int128)a2 * b[1] + (unsigned __int128)a3 * b[0];
    c  = (unsigned __int128)a4 * b[4];
    d += (unsigned __int128)R52 * (uint64_t)c;
    c >>= 64;
    t3 = (uint64_t)d & M52;
    d >>= 52;
}
