#pragma once
#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <array>
#include <algorithm>
#include <random>

namespace secp256k1 {
namespace cuda {

// ============================================================
// Host Helper Classes
// ============================================================

// Helper: Hex string to bytes
inline std::array<uint8_t, 32> hex_to_bytes(const char* hex) {
    std::array<uint8_t, 32> bytes{};
    size_t len = strlen(hex);
    if (len > 64) len = 64;
    
    char c;
    uint8_t val;
    size_t byte_idx;

    for (size_t i = 0; i < len; i++) {
        c = hex[i];
        val = 0;
        if (c >= '0' && c <= '9') val = c - '0';
        else if (c >= 'a' && c <= 'f') val = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') val = c - 'A' + 10;
        
        byte_idx = (len - 1 - i) / 2;
        if ((len - 1 - i) % 2 == 0) {
            bytes[31 - byte_idx] |= val;
        } else {
            bytes[31 - byte_idx] |= (val << 4);
        }
    }
    return bytes;
}

// Helper: Bytes to hex string
inline std::string bytes_to_hex(const uint8_t* bytes, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << (int)bytes[i];
    }
    return ss.str();
}

// Helper: Compare hex strings (case-insensitive)
inline bool hex_equal(const std::string& a, const char* b) {
    if (a.length() != strlen(b)) return false;
    char ca;
    char cb;
    for (size_t i = 0; i < a.length(); i++) {
        ca = a[i];
        cb = b[i];
        if (ca >= 'A' && ca <= 'F') ca += 32; // to lowercase
        if (cb >= 'A' && cb <= 'F') cb += 32;
        if (ca != cb) return false;
    }
    return true;
}

// Host Scalar Class
struct HostScalar {
    uint64_t limbs[4];

    HostScalar() { memset(limbs, 0, sizeof(limbs)); }

    static HostScalar from_bytes(const std::array<uint8_t, 32>& bytes) {
        HostScalar s;
        uint64_t limb;
        for (int i = 0; i < 4; ++i) {
            limb = 0;
            for (int j = 0; j < 8; ++j) {
                limb |= (uint64_t)bytes[31 - (i * 8 + j)] << (j * 8);
            }
            s.limbs[i] = limb;
        }
        return s;
    }

    static HostScalar from_hex(const char* hex) {
        return from_bytes(hex_to_bytes(hex));
    }
    
    static HostScalar from_uint64(uint64_t v) {
        HostScalar s;
        s.limbs[0] = v;
        return s;
    }
    
    static HostScalar zero() { return HostScalar(); }
    static HostScalar one() { return from_uint64(1); }

    Scalar to_device() const {
        Scalar s;
        for(int i=0; i<4; i++) s.limbs[i] = limbs[i];
        return s;
    }

    // Basic arithmetic on host for verification
    HostScalar operator+(const HostScalar& other) const {
        // Simple implementation using __int128
        uint64_t N[4] = {
            0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
        };
        
        HostScalar r;
        uint64_t carry = 0;
        for(int i=0; i<4; i++) {
            const uint64_t sum = limbs[i] + other.limbs[i];
            const uint64_t carry0 = (sum < limbs[i]) ? 1ULL : 0ULL;
            r.limbs[i] = sum + carry;
            carry = carry0 | ((r.limbs[i] < sum) ? 1ULL : 0ULL);
        }
        
        // Subtract N if needed
        bool ge = false;
        if (carry) ge = true;
        else {
            for(int i=3; i>=0; i--) {
                if(r.limbs[i] > N[i]) { ge = true; break; }
                if(r.limbs[i] < N[i]) break;
                if(i==0) ge = true; // equal
            }
        }
        
        if (ge) {
            uint64_t borrow = 0;
            for(int i=0; i<4; i++) {
                const uint64_t diff = r.limbs[i] - N[i];
                const uint64_t borrow0 = (r.limbs[i] < N[i]) ? 1ULL : 0ULL;
                r.limbs[i] = diff - borrow;
                borrow = borrow0 | ((diff < borrow) ? 1ULL : 0ULL);
            }
        }
        return r;
    }
    
    HostScalar operator-(const HostScalar& other) const {
        uint64_t N[4] = {
            0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
        };
        
        HostScalar r;
        uint64_t borrow = 0;
        for(int i=0; i<4; i++) {
            const uint64_t diff = limbs[i] - other.limbs[i];
            const uint64_t borrow0 = (limbs[i] < other.limbs[i]) ? 1ULL : 0ULL;
            r.limbs[i] = diff - borrow;
            borrow = borrow0 | ((diff < borrow) ? 1ULL : 0ULL);
        }
        
        if (borrow) {
            uint64_t carry = 0;
            for(int i=0; i<4; i++) {
                const uint64_t sum = r.limbs[i] + N[i];
                const uint64_t carry0 = (sum < r.limbs[i]) ? 1ULL : 0ULL;
                r.limbs[i] = sum + carry;
                carry = carry0 | ((r.limbs[i] < sum) ? 1ULL : 0ULL);
            }
        }
        return r;
    }

    HostScalar operator*(const HostScalar& other) const {
        // Double-and-add multiplication mod N
        HostScalar res; // 0
        HostScalar base = *this;
        
        for (int i = 0; i < 256; i++) {
            int limb = i / 64;
            int bit = i % 64;
            if ((other.limbs[limb] >> bit) & 1) {
                res = res + base;
            }
            base = base + base;
        }
        return res;
    }
    
    bool operator==(const HostScalar& other) const {
        for(int i=0; i<4; i++) if(limbs[i] != other.limbs[i]) return false;
        return true;
    }
};

// Host FieldElement Class
struct HostFieldElement {
    uint64_t limbs[4];
    
    HostFieldElement() { memset(limbs, 0, sizeof(limbs)); }

    HostFieldElement(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) {
        limbs[0] = l0; limbs[1] = l1; limbs[2] = l2; limbs[3] = l3;
    }
    
    static HostFieldElement from_bytes(const std::array<uint8_t, 32>& bytes) {
        HostFieldElement f;
        uint64_t limb;
        for (int i = 0; i < 4; ++i) {
            limb = 0;
            for (int j = 0; j < 8; ++j) {
                limb |= (uint64_t)bytes[31 - (i * 8 + j)] << (j * 8);
            }
            f.limbs[i] = limb;
        }
        return f;
    }
    
    static HostFieldElement from_uint64(uint64_t v) {
        HostFieldElement f;
        f.limbs[0] = v;
        return f;
    }
    
    static HostFieldElement zero() { return HostFieldElement(); }
    static HostFieldElement one() { return from_uint64(1); }
    
    std::array<uint8_t, 32> to_bytes() const {
        std::array<uint8_t, 32> bytes;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 8; ++j) {
                bytes[31 - (i * 8 + j)] = (limbs[i] >> (j * 8)) & 0xFF;
            }
        }
        return bytes;
    }
    
    std::string to_hex() const {
        return bytes_to_hex(to_bytes().data(), 32);
    }
    
    FieldElement to_device() const {
        FieldElement f;
        for(int i=0; i<4; i++) f.limbs[i] = limbs[i];
        return f;
    }
    
    static HostFieldElement from_device(const FieldElement& f) {
        HostFieldElement hf;
        for(int i=0; i<4; i++) hf.limbs[i] = f.limbs[i];
        return hf;
    }
    
    // Operations using CUDA kernels
    HostFieldElement operator+(const HostFieldElement& other) const {
        FieldElement d_a = to_device();
        FieldElement d_b = other.to_device();
        FieldElement d_r;
        
        FieldElement *d_a_ptr, *d_b_ptr, *d_r_ptr;
        cudaMalloc(&d_a_ptr, sizeof(FieldElement));
        cudaMalloc(&d_b_ptr, sizeof(FieldElement));
        cudaMalloc(&d_r_ptr, sizeof(FieldElement));
        
        cudaMemcpy(d_a_ptr, &d_a, sizeof(FieldElement), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_ptr, &d_b, sizeof(FieldElement), cudaMemcpyHostToDevice);
        
        field_add_kernel<<<1, 1>>>(d_a_ptr, d_b_ptr, d_r_ptr, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_r, d_r_ptr, sizeof(FieldElement), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a_ptr); cudaFree(d_b_ptr); cudaFree(d_r_ptr);
        return from_device(d_r);
    }
    
    HostFieldElement operator-(const HostFieldElement& other) const {
        FieldElement d_a = to_device();
        FieldElement d_b = other.to_device();
        FieldElement d_r;
        
        FieldElement *d_a_ptr, *d_b_ptr, *d_r_ptr;
        cudaMalloc(&d_a_ptr, sizeof(FieldElement));
        cudaMalloc(&d_b_ptr, sizeof(FieldElement));
        cudaMalloc(&d_r_ptr, sizeof(FieldElement));
        
        cudaMemcpy(d_a_ptr, &d_a, sizeof(FieldElement), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_ptr, &d_b, sizeof(FieldElement), cudaMemcpyHostToDevice);
        
        field_sub_kernel<<<1, 1>>>(d_a_ptr, d_b_ptr, d_r_ptr, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_r, d_r_ptr, sizeof(FieldElement), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a_ptr); cudaFree(d_b_ptr); cudaFree(d_r_ptr);
        return from_device(d_r);
    }
    
    HostFieldElement operator*(const HostFieldElement& other) const {
        FieldElement d_a = to_device();
        FieldElement d_b = other.to_device();
        FieldElement d_r;
        
        FieldElement *d_a_ptr, *d_b_ptr, *d_r_ptr;
        cudaMalloc(&d_a_ptr, sizeof(FieldElement));
        cudaMalloc(&d_b_ptr, sizeof(FieldElement));
        cudaMalloc(&d_r_ptr, sizeof(FieldElement));
        
        cudaMemcpy(d_a_ptr, &d_a, sizeof(FieldElement), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_ptr, &d_b, sizeof(FieldElement), cudaMemcpyHostToDevice);
        
        field_mul_kernel<<<1, 1>>>(d_a_ptr, d_b_ptr, d_r_ptr, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_r, d_r_ptr, sizeof(FieldElement), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a_ptr); cudaFree(d_b_ptr); cudaFree(d_r_ptr);
        return from_device(d_r);
    }
    
    HostFieldElement inverse() const {
        FieldElement d_a = to_device();
        FieldElement d_r;
        
        FieldElement *d_a_ptr, *d_r_ptr;
        cudaMalloc(&d_a_ptr, sizeof(FieldElement));
        cudaMalloc(&d_r_ptr, sizeof(FieldElement));
        
        cudaMemcpy(d_a_ptr, &d_a, sizeof(FieldElement), cudaMemcpyHostToDevice);
        
        field_inv_kernel<<<1, 1>>>(d_a_ptr, d_r_ptr, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_r, d_r_ptr, sizeof(FieldElement), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a_ptr); cudaFree(d_r_ptr);
        return from_device(d_r);
    }
    
    bool operator==(const HostFieldElement& other) const {
        for(int i=0; i<4; i++) if(limbs[i] != other.limbs[i]) return false;
        return true;
    }
};

// Helper kernel for normalization
__global__ void normalize_kernel(JacobianPoint* p) {
    if (p->infinity) return;
    FieldElement z_inv;
    field_inv(&p->z, &z_inv);
    FieldElement z_inv2;
    field_mul(&z_inv, &z_inv, &z_inv2);
    FieldElement z_inv3;
    field_mul(&z_inv2, &z_inv, &z_inv3);
    
    field_mul(&p->x, &z_inv2, &p->x);
    field_mul(&p->y, &z_inv3, &p->y);
    
    p->z.limbs[0] = 1; p->z.limbs[1] = 0; p->z.limbs[2] = 0; p->z.limbs[3] = 0;
}

// Host Point Class
struct HostPoint {
    HostFieldElement x_fe;
    HostFieldElement y_fe;
    HostFieldElement z_fe;
    bool infinity;

    HostPoint() : infinity(true) {}

    static HostPoint generator() {
        HostPoint p;
        p.infinity = false;
        p.x_fe.limbs[0] = 0x59F2815B16F81798ULL; p.x_fe.limbs[1] = 0x029BFCDB2DCE28D9ULL;
        p.x_fe.limbs[2] = 0x55A06295CE870B07ULL; p.x_fe.limbs[3] = 0x79BE667EF9DCBBACULL;
        p.y_fe.limbs[0] = 0x9C47D08FFB10D4B8ULL; p.y_fe.limbs[1] = 0xFD17B448A6855419ULL;
        p.y_fe.limbs[2] = 0x5DA4FBFC0E1108A8ULL; p.y_fe.limbs[3] = 0x483ADA7726A3C465ULL;
        p.z_fe.limbs[0] = 1; p.z_fe.limbs[1] = 0; p.z_fe.limbs[2] = 0; p.z_fe.limbs[3] = 0;
        return p;
    }
    
    static HostPoint from_affine(const HostFieldElement& x, const HostFieldElement& y) {
        HostPoint p;
        p.infinity = false;
        p.x_fe = x;
        p.y_fe = y;
        p.z_fe = HostFieldElement::one();
        return p;
    }
    
    static HostPoint infinity_point() {
        return HostPoint();
    }

    JacobianPoint to_device() const {
        JacobianPoint p;
        p.x = x_fe.to_device(); p.y = y_fe.to_device(); p.z = z_fe.to_device(); p.infinity = infinity;
        return p;
    }

    static HostPoint from_device(const JacobianPoint& p) {
        HostPoint hp;
        hp.x_fe = HostFieldElement::from_device(p.x);
        hp.y_fe = HostFieldElement::from_device(p.y);
        hp.z_fe = HostFieldElement::from_device(p.z);
        hp.infinity = p.infinity;
        return hp;
    }

    void normalize() {
        if (infinity) return;
        JacobianPoint d_p = to_device();
        JacobianPoint* d_ptr;
        cudaMalloc(&d_ptr, sizeof(JacobianPoint));
        cudaMemcpy(d_ptr, &d_p, sizeof(JacobianPoint), cudaMemcpyHostToDevice);
        
        normalize_kernel<<<1, 1>>>(d_ptr);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_p, d_ptr, sizeof(JacobianPoint), cudaMemcpyDeviceToHost);
        cudaFree(d_ptr);
        
        *this = from_device(d_p);
    }

    HostFieldElement x() const { 
        HostPoint copy = *this;
        copy.normalize();
        return copy.x_fe; 
    }
    
    HostFieldElement y() const { 
        HostPoint copy = *this;
        copy.normalize();
        return copy.y_fe; 
    }
    
    bool is_infinity() const { return infinity; }

    HostPoint scalar_mul(const HostScalar& k) const {
        JacobianPoint d_p = to_device();
        Scalar d_k = k.to_device();
        JacobianPoint d_r;
        
        JacobianPoint *d_p_ptr, *d_r_ptr;
        Scalar *d_k_ptr;
        
        cudaMalloc(&d_p_ptr, sizeof(JacobianPoint));
        cudaMalloc(&d_r_ptr, sizeof(JacobianPoint));
        cudaMalloc(&d_k_ptr, sizeof(Scalar));
        
        cudaMemcpy(d_p_ptr, &d_p, sizeof(JacobianPoint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k_ptr, &d_k, sizeof(Scalar), cudaMemcpyHostToDevice);
        
        scalar_mul_batch_kernel<<<1, 1>>>(d_p_ptr, d_k_ptr, d_r_ptr, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_r, d_r_ptr, sizeof(JacobianPoint), cudaMemcpyDeviceToHost);
        
        cudaFree(d_p_ptr); cudaFree(d_r_ptr); cudaFree(d_k_ptr);
        return from_device(d_r);
    }
    
    HostPoint add(const HostPoint& other) const {
        JacobianPoint d_a = to_device();
        JacobianPoint d_b = other.to_device();
        JacobianPoint d_r;
        
        JacobianPoint *d_a_ptr, *d_b_ptr, *d_r_ptr;
        cudaMalloc(&d_a_ptr, sizeof(JacobianPoint));
        cudaMalloc(&d_b_ptr, sizeof(JacobianPoint));
        cudaMalloc(&d_r_ptr, sizeof(JacobianPoint));
        
        cudaMemcpy(d_a_ptr, &d_a, sizeof(JacobianPoint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_ptr, &d_b, sizeof(JacobianPoint), cudaMemcpyHostToDevice);
        
        point_add_kernel<<<1, 1>>>(d_a_ptr, d_b_ptr, d_r_ptr, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_r, d_r_ptr, sizeof(JacobianPoint), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a_ptr); cudaFree(d_b_ptr); cudaFree(d_r_ptr);
        return from_device(d_r);
    }
    
    HostPoint dbl() const {
        JacobianPoint d_a = to_device();
        JacobianPoint d_r;
        
        JacobianPoint *d_a_ptr, *d_r_ptr;
        cudaMalloc(&d_a_ptr, sizeof(JacobianPoint));
        cudaMalloc(&d_r_ptr, sizeof(JacobianPoint));
        
        cudaMemcpy(d_a_ptr, &d_a, sizeof(JacobianPoint), cudaMemcpyHostToDevice);
        
        point_dbl_kernel<<<1, 1>>>(d_a_ptr, d_r_ptr, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&d_r, d_r_ptr, sizeof(JacobianPoint), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a_ptr); cudaFree(d_r_ptr);
        return from_device(d_r);
    }
    
    void dbl_inplace() {
        *this = dbl();
    }
    
    void next_inplace() {
        *this = add(generator());
    }
    
    HostPoint negate() const {
        HostPoint p = *this;
        if (p.infinity) return p;
        // Negate Y: p.y = -p.y
        p.y_fe = HostFieldElement::zero() - p.y_fe;
        return p;
    }
    
    std::vector<uint8_t> to_compressed() const {
        HostPoint p = *this;
        p.normalize();
        std::vector<uint8_t> res(33);
        auto x_bytes = p.x_fe.to_bytes();
        auto y_bytes = p.y_fe.to_bytes();
        res[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
        std::copy(x_bytes.begin(), x_bytes.end(), res.begin() + 1);
        return res;
    }
    
    std::vector<uint8_t> to_uncompressed() const {
        HostPoint p = *this;
        p.normalize();
        std::vector<uint8_t> res(65);
        auto x_bytes = p.x_fe.to_bytes();
        auto y_bytes = p.y_fe.to_bytes();
        res[0] = 0x04;
        std::copy(x_bytes.begin(), x_bytes.end(), res.begin() + 1);
        std::copy(y_bytes.begin(), y_bytes.end(), res.begin() + 33);
        return res;
    }
};

} // namespace cuda
} // namespace secp256k1
