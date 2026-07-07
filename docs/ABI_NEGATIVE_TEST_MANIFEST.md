# ABI Negative-Test Manifest

Generated: 2026-07-06T17:48:31.591047+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 200
- Blocking functions: 0
- Null rejection evidence: 200
- Zero-edge evidence: 194
- Invalid-content evidence: 196
- Success-smoke evidence: 200

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

