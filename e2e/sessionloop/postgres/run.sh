#!/usr/bin/env bash
# Run the same tests directly and through a real transaction pool; remove only
# this invocation's isolated Docker project and its disposable database volume.
set -euo pipefail
cd "$(dirname "$0")"
project="agentic-pg-e2e-$$"
compose=(docker compose --project-name "$project")
cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then "${compose[@]}" logs --tail=80; fi
  "${compose[@]}" down --volumes --remove-orphans
  exit "$status"
}
trap cleanup EXIT
"${compose[@]}" up --wait --wait-timeout 60 -d
direct_port=$("${compose[@]}" port postgres 5432)
pooled_port=$("${compose[@]}" port pgbouncer 5432)
export AGENTIC_POSTGRES_DSN="postgres://e2e:disposable-e2e@$direct_port/e2e?sslmode=disable"
unset AGENTIC_PGBOUNCER_ADMIN_DSN
go test -race -count="${AGENTIC_POSTGRES_COUNT:-1}" -timeout=180s -v . "$@"
export AGENTIC_POSTGRES_DSN="postgres://e2e:disposable-e2e@$pooled_port/e2e?sslmode=disable"
export AGENTIC_PGBOUNCER_ADMIN_DSN="postgres://e2e:disposable-e2e@$pooled_port/pgbouncer?sslmode=disable"
go test -race -count="${AGENTIC_POSTGRES_COUNT:-1}" -timeout=180s -v . "$@"
