#!/usr/bin/env bash
# ===========================================================================
# generate_sbom.sh — Software Bill of Materials generator
# ===========================================================================
# Outputs a CycloneDX 1.6 SBOM for UltrafastSecp256k1.
# The library has zero runtime dependencies (header-only std::), so the SBOM
# lists the component metadata, build-time tools, and test dependencies.
#
# Usage: ./scripts/generate_sbom.sh [output_file]
# Default output: sbom.cdx.json
#
# If `cyclonedx-cli` or `syft` is available, it will be used.
# Otherwise, generates a minimal valid CycloneDX JSON directly.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT="${1:-${ROOT_DIR}/sbom.cdx.json}"

VERSION=$(cat "${ROOT_DIR}/VERSION.txt" 2>/dev/null || echo "0.0.0-dev")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Try syft first (if available)
if command -v syft &>/dev/null; then
    echo "Using syft to generate SBOM..."
    syft dir:"${ROOT_DIR}" -o cyclonedx-json@1.6 > "${OUTPUT}"
    echo "SBOM written to: ${OUTPUT}"
    exit 0
fi

echo "Generating minimal CycloneDX SBOM..."

cat > "${OUTPUT}" <<SBOM_EOF
{
  "\$schema": "http://cyclonedx.org/schema/bom-1.6.schema.json",
  "bomFormat": "CycloneDX",
  "specVersion": "1.6",
  "serialNumber": "urn:uuid:$(cat /proc/sys/kernel/random/uuid 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())' 2>/dev/null || echo "00000000-0000-0000-0000-000000000000")",
  "version": 1,
  "metadata": {
    "timestamp": "${TIMESTAMP}",
    "tools": {
      "components": [
        {
          "type": "application",
          "name": "generate_sbom.sh",
          "version": "1.0.0"
        }
      ]
    },
    "component": {
      "type": "library",
      "name": "UltrafastSecp256k1",
      "version": "${VERSION}",
      "group": "com.github.shrec",
      "description": "High-performance secp256k1 elliptic curve cryptography library",
      "licenses": [
        {
          "license": {
            "id": "MIT"
          }
        }
      ],
      "purl": "pkg:github/shrec/UltrafastSecp256k1@v${VERSION}",
      "externalReferences": [
        {
          "type": "vcs",
          "url": "https://github.com/shrec/UltrafastSecp256k1"
        },
        {
          "type": "website",
          "url": "https://github.com/shrec/UltrafastSecp256k1"
        },
        {
          "type": "issue-tracker",
          "url": "https://github.com/shrec/UltrafastSecp256k1/issues"
        }
      ]
    }
  },
  "components": [
    {
      "type": "library",
      "name": "fastsecp256k1",
      "version": "${VERSION}",
      "description": "Core secp256k1 C++ library (field, scalar, point, ECDSA, Schnorr)",
      "scope": "required",
      "purl": "pkg:github/shrec/UltrafastSecp256k1@v${VERSION}#cpu"
    },
    {
      "type": "library",
      "name": "ufsecp",
      "version": "${VERSION}",
      "description": "C ABI shim for UltrafastSecp256k1 (libsecp256k1-compatible API)",
      "scope": "optional",
      "purl": "pkg:github/shrec/UltrafastSecp256k1@v${VERSION}#compat"
    },
    {
      "type": "library",
      "name": "libsecp256k1",
      "version": "0.6.0",
      "description": "Bitcoin Core secp256k1 library (test-time dependency only)",
      "scope": "excluded",
      "purl": "pkg:github/bitcoin-core/secp256k1@v0.6.0",
      "evidence": {
        "occurrences": [
          {
            "location": "CMakeLists.txt",
            "line": 0
          }
        ]
      }
    }
  ],
  "dependencies": [
    {
      "ref": "UltrafastSecp256k1",
      "dependsOn": []
    }
  ]
}
SBOM_EOF

echo "SBOM written to: ${OUTPUT}"
echo "Format: CycloneDX 1.6 JSON"
echo "Components: $(grep -c '"type":' "${OUTPUT}") entries"
