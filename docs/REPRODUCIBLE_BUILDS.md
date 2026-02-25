# Reproducible Builds

This document describes how to verify that UltrafastSecp256k1 release binaries are
reproducible -- i.e., the same source code produces byte-identical outputs regardless
of who builds it.

---

## Why Reproducible Builds Matter

For a cryptographic library, reproducible builds provide:

1. **Supply-chain integrity** -- users can verify that published binaries match the source
2. **Tamper detection** -- any modification to source or toolchain changes the output hash
3. **Trust minimization** -- no need to trust the build server; anyone can reproduce

---

## Quick Verification

### Docker (recommended)

```bash
# Build the verification image (runs two builds + compares)
docker build -f Dockerfile.reproducible -t uf-repro-check .

# Run the comparison
docker run --rm uf-repro-check
```

Exit code `0` = builds match. Exit code `1` = builds differ.

### Local Script

```bash
# Requires: cmake, ninja, g++ (or set CC/CXX env vars)
chmod +x scripts/verify_reproducible_build.sh
./scripts/verify_reproducible_build.sh
```

---

## How It Works

1. **Pin ALL inputs**: compiler version, base image digest, timestamps
2. **Build twice** from the same source tree with identical flags
3. **Hash all output artifacts** (`.a`, `.so`, `.so.*`)
4. **Compare hashes** -- any difference indicates non-reproducibility

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `SOURCE_DATE_EPOCH` | `1700000000` | Clamp embedded timestamps |
| `TZ` | `UTC` | Eliminate timezone variation |
| `LC_ALL` | `C` | Stable locale for sort/string ops |

### Build Flags

The reproducible build uses:
- `-DCMAKE_BUILD_TYPE=Release` (deterministic optimization)
- `-DSECP256K1_USE_ASM=ON` (production assembly paths)
- Ninja generator (deterministic build order)
- Pinned compiler version (g++-13)

---

## Release Artifact Verification

Every release includes:

| Artifact | Description |
|----------|-------------|
| `SHA256SUMS.txt` | SHA-256 hashes of all release files |
| SLSA provenance | Build attestation via `actions/attest-build-provenance` |
| Cosign signatures | Sigstore Keyless signing of release archives |
| CycloneDX SBOM | Software Bill of Materials (`sbom.cdx.json`) |

### Verify SHA-256 Checksums

```bash
# Download both the release and SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
```

### Verify Cosign Signature

```bash
# Install cosign: https://docs.sigstore.dev/cosign/installation/
cosign verify-blob \
  --certificate-identity-regexp='github.com/shrec/UltrafastSecp256k1' \
  --certificate-oidc-issuer='https://token.actions.githubusercontent.com' \
  --signature UltrafastSecp256k1-vX.Y.Z-linux-x64.tar.gz.sig \
  UltrafastSecp256k1-vX.Y.Z-linux-x64.tar.gz
```

### Verify SLSA Provenance

```bash
# Install slsa-verifier: https://github.com/slsa-framework/slsa-verifier
slsa-verifier verify-artifact \
  --provenance-path attestation.intoto.jsonl \
  --source-uri github.com/shrec/UltrafastSecp256k1 \
  UltrafastSecp256k1-vX.Y.Z-linux-x64.tar.gz
```

---

## SBOM (Software Bill of Materials)

Generate a CycloneDX 1.6 SBOM:

```bash
./scripts/generate_sbom.sh [output_file]
# Default: sbom.cdx.json
```

The SBOM lists:
- **fastsecp256k1** -- core C++ library (zero runtime dependencies)
- **ufsecp** -- C ABI shim (optional)
- **libsecp256k1 v0.6.0** -- test-time dependency only (excluded from runtime)

UltrafastSecp256k1 has **zero runtime dependencies** beyond the C++ standard library.

---

## Known Limitations

- **Windows MSVC builds** are not bit-reproducible across different Visual Studio versions
  due to PDB paths and timestamp embedding. Use MinGW/Clang for reproducibility on Windows.
- **macOS builds** may vary across Xcode versions due to SDK differences.
- **LTO builds** may not be reproducible across compiler minor versions.
- Assembly-optimized paths (BMI2/ADX) produce identical output on the same compiler version.

---

## CI Integration

The `release.yml` workflow:
1. Builds on pinned OS images (ubuntu-24.04, macos-14, windows-latest)
2. Generates `SHA256SUMS.txt` for all artifacts
3. Creates SLSA provenance attestation
4. Signs archives with Sigstore cosign (keyless)
5. Generates CycloneDX SBOM
6. Attaches all verification artifacts to the GitHub Release
