# ABI Negative-Test Manifest

Generated: 2026-07-10T17:33:16.207627+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 202
- Blocking functions: 0
- Null rejection evidence: 202
- Zero-edge evidence: 196
- Invalid-content evidence: 198
- Success-smoke evidence: 202

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

