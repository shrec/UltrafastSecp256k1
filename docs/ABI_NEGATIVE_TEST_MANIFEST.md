# ABI Negative-Test Manifest

Generated: 2026-11-04T22:06:59.526292+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 210
- Blocking functions: 0
- Null rejection evidence: 210
- Zero-edge evidence: 203
- Invalid-content evidence: 208
- Success-smoke evidence: 210

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

