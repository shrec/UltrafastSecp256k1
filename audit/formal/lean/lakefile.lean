import Lake
open Lake DSL

package SafeGCD where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib SafeGCD where
  srcDir := "."
