# ABI Negative-Test Manifest

Generated: 2026-07-22T22:47:43.801554+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 205
- Blocking functions: 0
- Null rejection evidence: 205
- Zero-edge evidence: 199
- Invalid-content evidence: 202
- Success-smoke evidence: 205

## Blocking Functions

| Function | Missing Checks | Header |
|----------|----------------|--------|
| *(none)* | | |

## Rule

Every exported `ufsecp_*` function should satisfy the hostile-caller quartet when the contract implies it:

1. `null_rejection`
2. `zero_edge`
3. `invalid_content`
4. `success_smoke`

