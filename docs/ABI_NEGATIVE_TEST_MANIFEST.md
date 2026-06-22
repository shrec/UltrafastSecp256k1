# ABI Negative-Test Manifest

Generated: 2026-06-22T15:45:16.811442+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 197
- Blocking functions: 0
- Null rejection evidence: 197
- Zero-edge evidence: 191
- Invalid-content evidence: 192
- Success-smoke evidence: 197

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

