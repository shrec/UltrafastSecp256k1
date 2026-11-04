# ABI Negative-Test Manifest

Generated: 2026-11-04T21:32:36.041012+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 209
- Blocking functions: 0
- Null rejection evidence: 209
- Zero-edge evidence: 202
- Invalid-content evidence: 207
- Success-smoke evidence: 209

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

