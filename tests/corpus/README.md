# Fuzz Regression Corpus

This directory holds **pinned inputs** that exercise known edge-cases in parsers
and protocol code. Every CI run replays these inputs to prevent regressions.

---

## Structure

```
tests/corpus/
├── README.md           (this file)
├── der/                DER signature edge-cases
│   └── *.bin           raw byte inputs
├── schnorr/            Schnorr signature edge-cases
│   └── *.bin
├── pubkey/             Public key parser edge-cases
│   └── *.bin
├── address/            Address generation edge-cases
│   └── inputs.json     JSON test vectors
├── bip32/              BIP-32 path parser edge-cases
│   └── paths.txt       one path per line
└── ffi/                FFI boundary edge-cases
    └── inputs.json     structured test vectors
```

## Adding a New Corpus Entry

1. Identify the crash/misbehavior input.
2. Save the minimal reproduction input as a `.bin` or `.json` file.
3. Name it descriptively: `<issue_number>_<short_description>.bin`.
4. Add a line to `MANIFEST.txt` documenting the input.
5. Add a corresponding test case in the relevant fuzz test.

## Replay

The deterministic fuzz tests use hardcoded seeds (`0xDEADBEEF`, `0xADD12E55`)
for reproducibility. Corpus files supplement this with known-bad inputs.

```bash
# Run full fuzz suite (includes corpus replay)
ctest --test-dir build -R fuzz
```
