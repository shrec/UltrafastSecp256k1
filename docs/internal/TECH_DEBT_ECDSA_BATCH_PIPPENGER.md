# Tech Debt: ECDSA Batch Verify — True Pippenger MSM

**თარიღი:** 2026-05-05
**პრიორიტეტი:** P0 (High Impact, ~30-50% throughput gain for n≥128)
**სტატუსი:** Deferred — API change required
**Ticket:** P0-1 (perf audit 2026-05-05)

---

## პრობლემა

`ecdsa_batch_verify` ([src/cpu/src/batch_verify.cpp:355](../src/cpu/src/batch_verify.cpp))
N-ჯერ ცალ-ცალკე `dual_scalar_mul_gen_point`-ს გამოძახება ხდება:

```cpp
for (std::size_t i = 0; i < n; ++i) {
    auto R_prime = Point::dual_scalar_mul_gen_point(u1, u2, entries[i].public_key);
    check R_prime.x mod n == r_i   // per-sig x-coord check
}
```

**Cost:** N × ~27µs = O(N) → linear scaling.

Schnorr batch verify-ი კი **ერთ** Pippenger MSM-ს იყენებს 2N წერტილზე: O(N/log N).

---

## რატომ ვერ გაკეთდება ახლა

### ფუნდამენტური შეზღუდვა

Schnorr batch-ი მუშაობს იმიტომ, რომ მთლიანი check **ლინეარულია**:

```
sum(a_i * s_i)*G + sum(-a_i*e_i*P_i) + sum(-a_i*R_i) == ∞
```

ყველა წევრი ჯამდება — ერთი Pippenger შედეგი `== ∞` გადამოწმება.

ECDSA check-ი **არ არის ლინეარული** — N ცალ-ცალკე x-კოორდინატი უნდა შემოწმდეს:

```
R'_i.x mod n == r_i     (for each i independently)
```

`sum(a_i * R'_i)` Pippenger-ით გამოთვლა გვაძლევს **წერტილების ჯამს**, არა
ინდივიდუალურ R'_i-ებს. ჯამის x-კოორდინატი ≠ ინდივიდუალური r_i-ების check.

### y-ambiguity პრობლემა

Recovery flag-ის გარეშე: r_i-ს შეესაბამება **ორი** კანდიდატი წერტილი
(y_even და y_odd). N=100 სიგნატურისთვის სწორი y-კომბინაცია 2^{-100}
ალბათობით გამოიცნობა random-ად. Unusable.

---

## გამოსავალი: Bellman Batch Equation

**ECDSA-ს ჭეშმარიტი batch verify** შესაძლებელია recovery parameter-ით.

### ალგებრული ტოლობა

სიგნატურა (r_i, s_i) valid ↔ `s_i * R_i = z_i*G + r_i*Q_i`

სადაც R_i = (r_i, y_i) — **სრული** nonce წერტილი recovery_id-ით.

Random weights a_i-ით batch check:

```
sum(a_i * s_i * R_i) - sum(a_i * z_i)*G - sum(a_i * r_i * Q_i) == ∞
```

ეს **ლინეარულია** — ერთი Pippenger 3N წერტილზე:
- N × R_i (scalars: a_i * s_i)
- 1 × G (scalar: -sum(a_i * z_i))
- N × Q_i (scalars: -a_i * r_i)

### False positive ალბათობა

Random a_i-ებით: `Pr[false positive] < 1/group_order ≈ 2^{-256}` — cryptographically negligible.

---

## საჭირო ცვლილებები

### 1. API გაფართოება

```cpp
// current:
struct ECDSABatchEntry {
    std::array<uint8_t, 32> msg_hash;
    Point public_key;
    ECDSASignature signature;
};

// needed:
struct ECDSABatchEntryV2 {
    std::array<uint8_t, 32> msg_hash;
    Point public_key;
    ECDSASignature signature;
    uint8_t recovery_id;   // 0 or 1 — y-parity of nonce point R
};
```

ABI ფაილები: `include/ufsecp/ufsecp.h`, `docs/API_REFERENCE.md`,
`docs/ABI_VERSIONING.md`.

### 2. Signing-ის recovery_id output

`ufsecp_ecdsa_sign` უნდა დაბრუნებდეს recovery_id-ს (1 ბიტი):

```c
ufsecp_error_t ufsecp_ecdsa_sign_ex(
    ufsecp_ctx* ctx,
    const uint8_t msg32[32],
    const uint8_t privkey32[32],
    uint8_t sig64_out[64],
    uint8_t* recovery_id_out   // NEW: 0 or 1
);
```

`ct_sign.cpp`-ში recovery_id-ი `R.y`-ს პარიტეტიდან გამომდინარეობს —
CT path-ს შიდა `k.y_parity` უკვე გამოითვლება.

### 3. Batch verify V2 იმპლემენტაცია

```cpp
// batch_verify.cpp
bool ecdsa_batch_verify_v2(const ECDSABatchEntryV2* entries, std::size_t n) {
    // 1. validate + batch-invert s_i (existing logic)

    // 2. lift r_i → R_i using recovery_id (new: sqrt not needed)
    //    R_i.x = r_i (parse as field element)
    //    R_i.y = recovered from (r_i, recovery_id) via curve equation
    //    lift_x + parity flip if recovery_id == 1

    // 3. accumulate into MSM:
    //    G_scalar = -sum(a_i * z_i * s_inv_i)
    //    R_i scalar = a_i * 1
    //    Q_i scalar = -a_i * r_i * s_inv_i
    //    where a_i = schnorr-style batch weight (hash-based)

    // 4. single Pippenger over G + {R_i}_{i<n} + {Q_i}_{i<n}
    //    == 1 + 2N points total

    // 5. check result == infinity
}
```

R_i-ს lift ხდება `lift_x_impl`-ის ანალოგიით (`schnorr.cpp:200-215` ნახე)
+ parity flip recovery_id-ის მიხედვით. O(1) per sig (sqrt ამოღება ერთხელ).

### 4. Old API backward compat

```cpp
// Keep ecdsa_batch_verify() unchanged — calls old N-loop path
// Add ecdsa_batch_verify_v2() — new Bellman path
// ABI: ufsecp_ecdsa_batch_verify_v2() with recovery_id array param
```

---

## შეფასება

| მეტრიკა | ახლა (N-loop) | V2 (Pippenger) |
|---------|--------------|----------------|
| n=16 | 16 × 27µs = 432µs | ~80µs (+overhead) |
| n=64 | 64 × 27µs = 1.7ms | ~240µs |
| n=128 | 128 × 27µs = 3.5ms | ~420µs |
| n=1024 | 1024 × 27µs = 27.6ms | ~2.5ms |
| scaling | O(N) | O(N/log N) |
| gain at n=128 | baseline | **~35-50%** |
| gain at n=1024 | baseline | **~55-65%** |

---

## დამოკიდებულებები

1. `ufsecp_ecdsa_sign_ex` — recovery_id output (CT path-ში უკვე გამოიანგარიშება)
2. `lift_x` helper — recovery_id + parity (schnorr.cpp-ის ანალოგი)
3. Bellman batch weight — `schnorr_challenge_scalar`-ის ანალოგი
4. ABI version bump — `UFSECP_ABI_VERSION` + docs
5. Tests: exploit test + audit_integration.cpp-ში batch verify parity test

---

## კავშირი არსებულ კოდთან

- `src/cpu/src/batch_verify.cpp:120-185` — Schnorr batch MSM (reference impl)
- `src/cpu/src/batch_verify.cpp:298-419` — ECDSA loop (replace with V2)
- `src/cpu/src/schnorr.cpp:200-215` — `lift_x_cached` (adapt for recovery)
- `src/cpu/src/ct_sign.cpp` — CT signing, y-parity already tracked internally
- `include/secp256k1/batch_verify.hpp` — `ECDSABatchEntry` struct

---

## ნოტები

- P0-3 (OpenCL double inversion) — false positive: Jacobian parity-სთვის
  inversion აუცილებელია secp256k1-ზე. არ გასწოროდეს.
- P0-1 ამ audit-სა და [perf_audit_2026_05_05.md](../../workingdocs/perf_audit_2026-05-05.md)-ში.
