// Micro-benchmark for atomic-level operations
// Point Add, Point Double, Field Multiply, Field Square, Field Inverse
// These are the REAL building blocks of ECC performance

#include "secp256k1/fast.hpp"
#include "secp256k1/selftest.hpp"
#include "secp256k1/init.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cstdlib>

using namespace secp256k1::fast;
using namespace std::chrono;

// High-resolution benchmark helper
template<typename Func>
double benchmark_ns(Func&& f, int iterations = 10000) {
    // Warmup
    for (int i = 0; i < 100; ++i) f();
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f();
    }
    auto end = high_resolution_clock::now();
    
    double total_ns = static_cast<double>(duration_cast<nanoseconds>(end - start).count());
    return total_ns / iterations;
}

int main() {
    // Force stdout flush for MinGW/Clang compatibility
    std::cout.sync_with_stdio(true);
    std::cout << std::unitbuf;  // Auto-flush after every output
    
    // Run 29-step validation (now works on all compilers!)
    if (!secp256k1::fast::ensure_library_integrity(true)) {
        std::cerr << "[FATAL] Library validation failed!\n";
        return 1;
    }
    
    // Set cache directory
#ifdef _WIN32
    _putenv("SECP256K1_CACHE_DIR=F:\\EccTables");
#else
    setenv("SECP256K1_CACHE_DIR", "/tmp/EccTables", 1);
#endif

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  SECP256K1-FAST: Atomic Operations Micro-Benchmark\n";
    std::cout << "  Testing individual building blocks (10,000 iterations each)\n";
    std::cout << "================================================================\n\n";

    // Prepare test data -- full 256-bit values for representative results
    Point G = Point::generator();
    (void)G;
    Point P = scalar_mul_generator(Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35"));
    Point Q = scalar_mul_generator(Scalar::from_hex(
        "7c076ff316692a3d7eb3c3bb0f8b1488cf72e1afcd929e29307032997a838a3d"));
    
    FieldElement a = FieldElement::from_hex(
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    FieldElement b = FieldElement::from_hex(
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");
    FieldElement c = FieldElement::from_hex(
        "9c47d08ffb10d4b8483ada7726a3c46555a06295ce870b0779be667ef9dcbbac");
    (void)c;

    std::cout << "=== POINT OPERATIONS (Jacobian Coordinates) ===\n\n";
    
    // Point Addition: Jacobian + Jacobian (immutable - with allocation)
    double point_add_immutable = benchmark_ns([&]() {
        volatile auto R = P.add(Q);
        (void)R;
    });
    std::cout << "Point Add (immutable):       " << std::fixed << std::setprecision(2) 
              << point_add_immutable << " ns/op  [12M+4S + allocation]\n";
    
    // Point Addition: In-place (mutable - no allocation!)
    Point P_copy = P;
    double point_add_inplace = benchmark_ns([&]() {
        P_copy = P;  // Reset for each iteration
        P_copy.add_inplace(Q);
    }, 1000);  // Fewer iterations since we modify
    std::cout << "Point Add (in-place):        " << std::fixed << std::setprecision(2) 
              << point_add_inplace << " ns/op  [12M+4S, no allocation]\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << (point_add_immutable / point_add_inplace) << "x\n\n";

    // Use the in-place timing as the representative Jac+Jac add cost
    double point_add_jac_jac = point_add_inplace;
    
    // Point Doubling: Immutable
    double point_double_immutable = benchmark_ns([&]() {
        volatile auto R = P.dbl();
        (void)R;
    });
    std::cout << "Point Double (immutable):    " << std::fixed << std::setprecision(2) 
              << point_double_immutable << " ns/op  [4M+4S + allocation]\n";
    
    // Point Doubling: In-place
    double point_dbl_inplace = benchmark_ns([&]() {
        P_copy = P;  // Reset
        P_copy.dbl_inplace();
    }, 1000);
    std::cout << "Point Double (in-place):     " << std::fixed << std::setprecision(2) 
              << point_dbl_inplace << " ns/op  [4M+4S, no allocation]\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << (point_double_immutable / point_dbl_inplace) << "x\n\n";

    // Use the in-place timing as the representative double cost
    double point_double_time = point_dbl_inplace;
    
    // Next (G+1): Immutable vs In-place
    double next_immutable = benchmark_ns([&]() {
        volatile auto R = P.next();
        (void)R;
    });
    std::cout << "Next G+1 (immutable):        " << std::fixed << std::setprecision(2) 
              << next_immutable << " ns/op  [mixed 7M+4S + allocation]\n";
    
    double next_inplace = benchmark_ns([&]() {
        P_copy = P;  // Reset
        P_copy.next_inplace();
    }, 1000);
    std::cout << "Next G+1 (in-place):         " << std::fixed << std::setprecision(2) 
              << next_inplace << " ns/op  [mixed 7M+4S, no allocation]\n";
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << (next_immutable / next_inplace) << "x\n";
    
    // Point Negation (trivial - just flips Y coordinate)
    double point_negate_time = benchmark_ns([&]() {
        volatile auto R = P.negate();
        (void)R;
    });
    std::cout << "Point Negation (-P):         " << std::fixed << std::setprecision(2) 
              << point_negate_time << " ns/op  [trivial: Y := -Y]\n";

    std::cout << "\n=== FIELD OPERATIONS (256-bit modular arithmetic) ===\n\n";
    
    // Field Multiplication
    double field_mul_time = benchmark_ns([&]() {
        volatile auto r = a * b;
        (void)r;
    });
    std::cout << "Field Multiplication (a*b):  " << std::fixed << std::setprecision(2) 
              << field_mul_time << " ns/op\n";
    
    // Field Squaring
    double field_sqr_time = benchmark_ns([&]() {
        volatile auto r = a.square();
        (void)r;
    });
    std::cout << "Field Squaring (a^2):        " << std::fixed << std::setprecision(2) 
              << field_sqr_time << " ns/op\n";
    
    // Field Addition
    double field_add_time = benchmark_ns([&]() {
        volatile auto r = a + b;
        (void)r;
    });
    std::cout << "Field Addition (a+b):        " << std::fixed << std::setprecision(2) 
              << field_add_time << " ns/op\n";
    
    // Field Subtraction
    double field_sub_time = benchmark_ns([&]() {
        volatile auto r = a - b;
        (void)r;
    });
    std::cout << "Field Subtraction (a-b):     " << std::fixed << std::setprecision(2) 
              << field_sub_time << " ns/op\n";

    // Field Inverse (expensive!)
    double field_inv_time = benchmark_ns([&]() {
        volatile auto r = a.inverse();
        (void)r;
    }, 1000);  // Only 1000 iterations - inverse is slow!
    std::cout << "Field Inverse (1/a):         " << std::fixed << std::setprecision(2) 
              << field_inv_time << " ns/op  [Fermat's little theorem: a^(p-2)]\n";

    std::cout << "\n=== COMPOSITE OPERATIONS ===\n\n";
    
    // Triple (3*P = P + P + P)
    double point_triple_time = benchmark_ns([&]() {
        volatile auto R = P.dbl().add(P);
        (void)R;
    });
    std::cout << "Point Triple (3*P):          " << std::fixed << std::setprecision(2) 
              << point_triple_time << " ns/op  [= 2*P + P]\n";
    
    // Jacobian -> Affine conversion (requires 1 inverse + 2 multiplications)
    // NOTE: benchmark the conversion itself, NOT scalar_mul (pre-compute the point)
    Point P_jac = scalar_mul_generator(Scalar::from_hex(
        "b5037ebecae0da656179c623f6cb73641db2aa0fabe888ffb78466fa18470379"));
    double to_affine_time = benchmark_ns([&]() {
        volatile auto x = P_jac.x();
        volatile auto y = P_jac.y();
        (void)x; (void)y;
    }, 1000);
    std::cout << "To Affine Conversion:        " << std::fixed << std::setprecision(2) 
              << to_affine_time << " ns/op  [1 inverse + 2-3 mul]\n";

    std::cout << "\n=== PERFORMANCE ANALYSIS ===\n\n";
    
    // Cost breakdown of Point Addition (Jacobian+Jacobian)
    double point_add_jac_theory = 12 * field_mul_time + 4 * field_sqr_time + 8 * field_add_time;
    std::cout << "Point Add Jac+Jac theoretical cost (12M+4S+8A):\n";
    std::cout << "  Estimated: " << std::fixed << std::setprecision(2) 
              << point_add_jac_theory << " ns\n";
    std::cout << "  Actual:    " << std::fixed << std::setprecision(2) 
              << point_add_jac_jac << " ns\n";
    std::cout << "  Overhead:  " << std::fixed << std::setprecision(1) 
              << ((point_add_jac_jac / point_add_jac_theory - 1.0) * 100.0) << "%\n\n";
    
    // TODO: Add mixed addition analysis when implemented
    // // Cost breakdown of Point Addition (Jacobian+Affine mixed)
    // double point_add_mixed_theory = 7 * field_mul_time + 4 * field_sqr_time + 9 * field_add_time;
    // std::cout << "Point Add Jac+Affine theoretical cost (7M+4S+9A):\n";
    // std::cout << "  Estimated: " << std::fixed << std::setprecision(2) 
    //           << point_add_mixed_theory << " ns\n";
    // std::cout << "  Actual:    " << std::fixed << std::setprecision(2) 
    //           << point_add_mixed << " ns\n";
    // std::cout << "  Overhead:  " << std::fixed << std::setprecision(1) 
    //           << ((point_add_mixed / point_add_mixed_theory - 1.0) * 100.0) << "%\n";
    // std::cout << "  Speedup vs Jac+Jac: " << std::fixed << std::setprecision(2)
    //           << (point_add_jac_jac / point_add_mixed) << "x\n\n";
    
    // Cost breakdown of Point Doubling
    double point_dbl_theory = 4 * field_mul_time + 4 * field_sqr_time + 4 * field_add_time;
    std::cout << "Point Doubling theoretical cost (4M + 4S + 4A):\n";
    std::cout << "  Estimated: " << std::fixed << std::setprecision(2) 
              << point_dbl_theory << " ns\n";
    std::cout << "  Actual:    " << std::fixed << std::setprecision(2) 
              << point_double_time << " ns\n";
    std::cout << "  Overhead:  " << std::fixed << std::setprecision(1) 
              << ((point_double_time / point_dbl_theory - 1.0) * 100.0) << "%\n\n";
    
    // Field operation ratios
    std::cout << "Field operation cost ratios (S = squaring, M = multiplication):\n";
    std::cout << "  S/M ratio: " << std::fixed << std::setprecision(3) 
              << (field_sqr_time / field_mul_time) << " (ideal: ~0.80)\n";
    std::cout << "  A/M ratio: " << std::fixed << std::setprecision(3) 
              << (field_add_time / field_mul_time) << " (ideal: ~0.05)\n";
    std::cout << "  I/M ratio: " << std::fixed << std::setprecision(1) 
              << (field_inv_time / field_mul_time) << "x (inverse is EXPENSIVE!)\n\n";

    std::cout << "=== KEY TAKEAWAYS ===\n\n";
    std::cout << "1. Field Inverse is " << std::fixed << std::setprecision(0) 
              << (field_inv_time / field_mul_time) << "x slower than multiplication\n";
    std::cout << "   -> Avoid inversions! Use Jacobian coordinates (no inversions needed)\n\n";
    
    std::cout << "2. Point operations are dominated by field multiplications:\n";
    std::cout << "   Point Add (Jac+Jac): ~" << std::fixed << std::setprecision(0) 
              << (point_add_jac_jac / field_mul_time) << "x field_mul\n";
    // TODO: Add when mixed addition is implemented
    // std::cout << "   Point Add (Mixed):   ~" << std::fixed << std::setprecision(0) 
    //           << (point_add_mixed / field_mul_time) << "x field_mul\n";
    std::cout << "   Point Double: ~" << std::fixed << std::setprecision(0) 
              << (point_double_time / field_mul_time) << "x field_mul\n\n";
    
    std::cout << "3. To optimize point operations, focus on:\n";
    std::cout << "   [A] Faster field multiplication (ASM, BMI2, MULX)\n";
    std::cout << "   [B] Reduce number of field ops (better formulas)\n";
    std::cout << "   [C] Avoid Jacobian->Affine conversions (expensive inverse!)\n\n";

    std::cout << "================================================================\n\n";

    return 0;
}
