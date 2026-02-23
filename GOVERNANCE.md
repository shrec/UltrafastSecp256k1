# Governance Model

## Overview

UltrafastSecp256k1 follows a **Benevolent Dictator For Life (BDFL)** governance model, common among small-to-medium open-source cryptographic projects. All final technical and project decisions rest with the lead maintainer.

## Roles

### Lead Maintainer (BDFL)

- **Current**: [@shrec](https://github.com/shrec)
- **Responsibilities**:
  - Final authority on all project decisions (architecture, releases, merges)
  - Reviews and merges pull requests
  - Manages releases and versioning
  - Sets project direction and roadmap
  - Triages issues and security reports
  - Maintains CI/CD infrastructure

### Contributors

Anyone who submits a pull request, files an issue, or improves documentation. Contributors:

- Must follow the [Contributing Guidelines](CONTRIBUTING.md)
- Must sign off commits per the [DCO](CONTRIBUTING.md#-developer-certificate-of-origin-dco)
- Must adhere to the [Coding Standards](docs/CODING_STANDARDS.md)
- Have no merge authority; all PRs require maintainer approval

### Security Reporters

Individuals who report vulnerabilities through [GitHub Security Advisories](https://github.com/shrec/UltrafastSecp256k1/security/advisories) as described in [SECURITY.md](SECURITY.md). Reports are handled privately by the lead maintainer.

## Decision Process

1. **Routine changes** (bug fixes, documentation, minor improvements): The lead maintainer reviews and merges directly.
2. **Significant changes** (new features, API changes, architecture): Discussed in a GitHub issue or PR before implementation. The lead maintainer makes the final call.
3. **Cryptographic changes** (math modifications, algorithm changes): Require explicit approval from the lead maintainer. No math changes without review. No weakening of search coverage or correctness guarantees is permitted.
4. **Security decisions**: Handled privately by the lead maintainer per [SECURITY.md](SECURITY.md).

## Release Process

- The lead maintainer decides when to cut a release.
- Releases follow [Semantic Versioning](https://semver.org/).
- All CI checks must pass before a release is tagged.
- Release assets are built and published via GitHub Actions ([release.yml](.github/workflows/release.yml)).

## Amendments

This governance model may be updated by the lead maintainer. Significant governance changes will be documented in the commit history and announced in the relevant release notes.

## Contact

- **Issues**: [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues)
- **Security**: [SECURITY.md](SECURITY.md)
- **Contributions**: [CONTRIBUTING.md](CONTRIBUTING.md)
