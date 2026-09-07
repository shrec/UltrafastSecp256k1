#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "secp256k1/detail/arith64.hpp"

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::uint64_t a = std::strtoull(argv[1], nullptr, 16);
    const std::uint64_t b = std::strtoull(argv[2], nullptr, 16);
#ifdef __SIZEOF_INT128__
    std::cout << "__SIZEOF_INT128__=" << __SIZEOF_INT128__ << ' ';
#else
    std::cout << "__SIZEOF_INT128__=undefined ";
#endif
#ifdef SECP256K1_NO_INT128
    std::cout << "SECP256K1_NO_INT128=defined ";
#else
    std::cout << "SECP256K1_NO_INT128=undefined ";
#endif
#ifdef _MSC_VER
    std::cout << "_MSC_VER=defined\n";
#else
    std::cout << "_MSC_VER=undefined\n";
#endif
    std::cout << std::hex << std::setfill('0')
              << "a=" << std::setw(16) << a
              << " b=" << std::setw(16) << b
              << " high=" << std::setw(16)
              << secp256k1::detail::mulhi64(a, b) << '\n';
}
