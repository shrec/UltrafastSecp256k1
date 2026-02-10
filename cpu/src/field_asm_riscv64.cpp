// RISC-V 64-bit Assembly Wrappers for Field Operations
// Provides C++ interface to RISC-V assembly optimizations

#ifdef SECP256K1_HAS_RISCV_ASM

#include <secp256k1/field.hpp>
#include <cstdint>

namespace secp256k1::fast {

// External RISC-V assembly functions (defined in field_asm_riscv64.S)
extern "C" {
    void field_mul_asm_riscv64(uint64_t* r, const uint64_t* a, const uint64_t* b);
    void field_square_asm_riscv64(uint64_t* r, const uint64_t* a);
    void field_add_asm_riscv64(uint64_t* r, const uint64_t* a, const uint64_t* b);
    void field_sub_asm_riscv64(uint64_t* r, const uint64_t* a, const uint64_t* b);
    void field_negate_asm_riscv64(uint64_t* r, const uint64_t* a);
}

// Multiplication wrapper
FieldElement field_mul_riscv(const FieldElement& a, const FieldElement& b) {
    FieldElement result;
    field_mul_asm_riscv64(
        const_cast<uint64_t*>(result.limbs().data()),
        a.limbs().data(),
        b.limbs().data()
    );
    return result;
}

// Squaring wrapper
FieldElement field_square_riscv(const FieldElement& a) {
    FieldElement result;
    field_square_asm_riscv64(
        const_cast<uint64_t*>(result.limbs().data()),
        a.limbs().data()
    );
    return result;
}

// Addition wrapper
FieldElement field_add_riscv(const FieldElement& a, const FieldElement& b) {
    FieldElement result;
    field_add_asm_riscv64(
        const_cast<uint64_t*>(result.limbs().data()),
        a.limbs().data(),
        b.limbs().data()
    );
    return result;
}

// Subtraction wrapper
FieldElement field_sub_riscv(const FieldElement& a, const FieldElement& b) {
    FieldElement result;
    field_sub_asm_riscv64(
        const_cast<uint64_t*>(result.limbs().data()),
        a.limbs().data(),
        b.limbs().data()
    );
    return result;
}

// Negation wrapper
FieldElement field_negate_riscv(const FieldElement& a) {
    FieldElement result;
    field_negate_asm_riscv64(
        const_cast<uint64_t*>(result.limbs().data()),
        a.limbs().data()
    );
    return result;
}


} // namespace secp256k1::fast

// Scalar arithmetic (outside namespace for C linkage compatibility)
extern "C" {
    void scalar_add_asm_riscv64(uint64_t* r, const uint64_t* a, const uint64_t* b);
    void scalar_sub_asm_riscv64(uint64_t* r, const uint64_t* a, const uint64_t* b);
}

namespace secp256k1::fast {

// Scalar add wrapper (for GLV decomposition) - mod N arithmetic
void scalar_add_riscv(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    scalar_add_asm_riscv64(r, a, b);
}

// Scalar sub wrapper (for GLV decomposition) - mod N arithmetic
void scalar_sub_riscv(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    scalar_sub_asm_riscv64(r, a, b);
}

} // namespace secp256k1::fast

#endif // SECP256K1_HAS_RISCV_ASM
