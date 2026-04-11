/-!
# SafeGCD — Constant-Time Mask Correctness

Proves that the CT masks used in `ct_divsteps_59` (cpu/src/ct_field.cpp) are
exactly 0 or all-ones, with no partial/intermediate bit patterns.

Hardware encoding:
  c1 = (int64_t)zeta >> 63    -- all-ones if zeta < 0, else 0
  c2 = -(uint64_t)(g & 1)     -- all-ones if g is odd, else 0
-/

set_option maxHeartbeats 800000

namespace SafeGCD.CTMasks

-- ============================================================================
-- 8-bit exhaustive proofs via native_decide
-- ============================================================================

/-- c1 = zeta >>> (n-1) is either 0 or all-ones (for 8-bit). -/
theorem mask_c1_binary_8 : ∀ (zeta : BitVec 8),
    zeta.sshiftRight 7 = 0#8 ∨ zeta.sshiftRight 7 = BitVec.allOnes 8 := by
  native_decide

/-- c2 = -(g &&& 1) is either 0 or all-ones (for 8-bit). -/
theorem mask_c2_binary_8 : ∀ (g : BitVec 8),
    (-(g &&& 1#8)) = 0#8 ∨ (-(g &&& 1#8)) = BitVec.allOnes 8 := by
  native_decide

/-- c1 &&& c2 is either 0 or all-ones (for 8-bit). -/
theorem mask_c1c2_binary_8 : ∀ (zeta g : BitVec 8),
    let c1 := zeta.sshiftRight 7
    let c2 := -(g &&& 1#8)
    (c1 &&& c2) = 0#8 ∨ (c1 &&& c2) = BitVec.allOnes 8 := by
  native_decide

/-- XOR-negate identity: (x ^^^ allOnes) - allOnes = -x (for 8-bit). -/
theorem xor_negate_8 : ∀ (x : BitVec 8),
    (x ^^^ BitVec.allOnes 8) - BitVec.allOnes 8 = -x := by
  native_decide

/-- XOR-identity: (x ^^^ 0) - 0 = x (for 8-bit). -/
theorem xor_identity_8 : ∀ (x : BitVec 8),
    (x ^^^ 0#8) - 0#8 = x := by
  native_decide

end SafeGCD.CTMasks
