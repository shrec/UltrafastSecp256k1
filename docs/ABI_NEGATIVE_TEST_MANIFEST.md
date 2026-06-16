# ABI Negative-Test Manifest

Generated: 2026-06-13T20:44:06.721551+00:00

Machine-generated hostile-caller coverage manifest for the public `ufsecp_*` ABI.

## Summary

- Exported functions scanned: 193
- Blocking functions: 0
- Null rejection evidence: 193
- Zero-edge evidence: 187
- Invalid-content evidence: 188
- Success-smoke evidence: 193

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

