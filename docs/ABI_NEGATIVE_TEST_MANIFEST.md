# ABI Negative-Test Manifest

Generated: 2026-06-17T16:46:49.997914+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 195
- Blocking functions: 0
- Null rejection evidence: 195
- Zero-edge evidence: 189
- Invalid-content evidence: 190
- Success-smoke evidence: 195

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

