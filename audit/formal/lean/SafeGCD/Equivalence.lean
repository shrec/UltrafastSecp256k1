/-!
# SafeGCD — CT ≡ Branching Equivalence

Proves that the branchless (constant-time) divstep implementation produces
identical results to the branching reference for all inputs.

This corresponds to THEOREM 7 in the Z3 proof (`safegcd_z3_proof.py`).
Verified at 8-bit width via `native_decide` (exhaustive, 2^24 input space).
-/

set_option maxHeartbeats 4000000

namespace SafeGCD.Equivalence

/-- Branching reference divstep (pure BitVec, 8-bit). -/
def divstep_ref (zeta f g : BitVec 8) : BitVec 8 × BitVec 8 × BitVec 8 :=
  let g_odd := (g &&& 1#8) == 1#8
  let zeta_neg := zeta.sshiftRight 7 == BitVec.allOnes 8
  let swap := zeta_neg && g_odd
  let f' := if swap then g else f
  let g_sum := if swap then g - f
               else if g_odd then g + f
               else g
  let g' := g_sum.ushiftRight 1
  let zeta' := if swap then -(zeta) - 2#8
               else zeta - 1#8
  (zeta', f', g')

/-- CT (branchless) divstep — mirrors ct_divsteps_59 mask logic, 8-bit. -/
def divstep_ct (zeta f g : BitVec 8) : BitVec 8 × BitVec 8 × BitVec 8 :=
  let c1 := zeta.sshiftRight 7
  let c2 := -(g &&& 1#8)
  let x  := (f ^^^ c1) - c1
  let g1 := g + (x &&& c2)
  let c1c2 := c1 &&& c2
  let zeta' := (zeta ^^^ c1c2) - 1#8
  let f' := f + (g1 &&& c1c2)
  let g' := g1.ushiftRight 1
  (zeta', f', g')

/-- Full equivalence: all three outputs match for all 8-bit inputs when f is odd. -/
theorem ct_equiv_ref_8 : ∀ (zeta f g : BitVec 8),
    (f &&& 1#8) == 1#8 →
    divstep_ct zeta f g = divstep_ref zeta f g := by
  native_decide

end SafeGCD.Equivalence
