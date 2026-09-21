#!/usr/bin/env bash
# Negative fixtures ensure the axiom audit really rejects unsound proof escapes.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Inject invalid proofs only into an ephemeral stdin compilation. They never
# become part of the library or its accepted proof environment.
expect_rejected() {
    local declaration="$1"
    local diagnostic="$2"
    local output
    if output=$(
        {
            sed '/^#audit_session_contract$/d' Audit.lean
            printf '%s\n' "$declaration" '#audit_session_contract'
        } | lake env lean --stdin 2>&1
    ); then
        printf '%s\n' 'ERROR: axiom audit accepted an invalid fixture.' >&2
        exit 1
    fi
    if [[ "$output" != *"$diagnostic"* ]]; then
        printf '%s\n' "$output" >&2
        printf '%s\n' 'ERROR: fixture failed for an unexpected reason.' >&2
        exit 1
    fi
}

expect_rejected \
    'theorem SessionContract.auditIncomplete : True := by sorry' \
    'depends on forbidden axiom sorryAx'

expect_rejected \
    $'axiom AuditFixture.untrusted : False\ntheorem SessionContract.auditTransitive : False := AuditFixture.untrusted' \
    'depends on forbidden axiom AuditFixture.untrusted'

printf '%s\n' 'Audit regressions passed: admitted proofs and transitive custom axioms rejected.'
