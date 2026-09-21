#!/usr/bin/env bash
# Build the specification, reject proof escapes, test the audit, and replay terms.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

lake build
# Re-elaborate the audit even if Lake's build artifacts are cached. It checks
# transitive dependencies, including admitted proofs in imported declarations.
lake env lean -DwarningAsError=true Audit.lean
bash audit-tests.sh
lake env leanchecker SessionContract
