# ABI Negative-Test Manifest

Generated: 2026-04-28T08:16:31.261078+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 180
- Blocking functions: 0
- Null rejection evidence: 180
- Zero-edge evidence: 168
- Invalid-content evidence: 175
- Success-smoke evidence: 180

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

