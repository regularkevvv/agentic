#!/usr/bin/env bash
# Exercise the exact published module revision in a fresh consumer, never the
# checkout's workspace or a previous hard-coded release. The optional revision
# supports rehearsing the same check against a pushed commit before tagging.
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
kind=${1:?usage: verify-release.sh root|harness|sessionloop|realtime [revision]}
case "$kind" in
  root) module=github.com/regularkevvv/agentic; prefix= ;;
  harness) module=github.com/regularkevvv/agentic/harness; prefix=harness/ ;;
  sessionloop) module=github.com/regularkevvv/agentic/harness/sessionloop; prefix=harness/sessionloop/ ;;
  realtime) module=github.com/regularkevvv/agentic/realtime; prefix=realtime/ ;;
  *) printf 'Unsupported release module: %s\n' "$kind" >&2; exit 2 ;;
esac

revision=${2:-${GITHUB_SHA:-}}
if [[ $# -lt 2 && ${GITHUB_REF_TYPE:-} == tag ]]; then
  case "$GITHUB_REF_NAME" in
    "${prefix}"v*) revision=${GITHUB_REF_NAME#"$prefix"} ;;
    *) printf 'Tag %s does not identify %s\n' "$GITHUB_REF_NAME" "$module" >&2; exit 2 ;;
  esac
fi
[[ -n "$revision" ]] || { printf 'A published revision is required\n' >&2; exit 2; }

consumer_dir=$(mktemp -d "${TMPDIR:-/tmp}/agentic-${kind}-consumer.XXXXXX")
trap 'rm -r -- "$consumer_dir"' EXIT
export GOWORK=off GOPROXY=direct GONOSUMDB="$module"
cd "$consumer_dir"
go mod init example.com/agentic-release-consumer
go get "$module@$revision"
expected=$(go list -m -f '{{.Version}}' "$module@$revision")
if [[ "$revision" == v* && "$expected" != "$revision" ]]; then
  printf 'Requested %s but resolved %s\n' "$revision" "$expected" >&2
  exit 1
fi

case "$kind" in
  root)
    go test -mod=mod -race -count=1 -timeout=60s "$module/..."
    ;;
  harness)
    cp "$repo_dir/.github/testdata/harness-consumer/main.go" .
    cp -R "$repo_dir/e2e/sessionloop" ./sessionloop
    go run -mod=mod .
    go test -mod=mod -race -count=1 -timeout=60s ./sessionloop
    # The released Harness must resolve its declared protocol and driver, not
    # accidental upgrades left behind in a consumer template.
    for dependency in github.com/regularkevvv/agentic github.com/regularkevvv/agentic/harness/sessionloop; do
      declared=$(go mod edit -json "$repo_dir/harness/go.mod" | jq -r --arg path "$dependency" '.Require[] | select(.Path == $path) | .Version')
      test -n "$declared"
      test "$(go list -m -f '{{.Version}}' "$dependency")" = "$declared"
    done
    ;;
  sessionloop)
    cp "$repo_dir/harness/sessionloop/testdata/consumer/main.go" .
    go run -mod=mod .
    test "$(go list -m all | wc -l | tr -d ' ')" = 2
    ;;
  realtime)
    cp "$repo_dir/realtime/testdata/consumer/main.go" .
    go run -mod=mod .
    # The consumer, realtime, and the protocol it declares: nothing else.
    test "$(go list -m all | wc -l | tr -d ' ')" = 3
    declared=$(go mod edit -json "$repo_dir/realtime/go.mod" | jq -r '.Require[] | select(.Path == "github.com/regularkevvv/agentic/harness/sessionloop") | .Version')
    test "$(go list -m -f '{{.Version}}' github.com/regularkevvv/agentic/harness/sessionloop)" = "$declared"
    ;;
esac

test -z "$(go list -m -f '{{if .Replace}}{{.Path}}{{end}}' all)"
test "$(go list -m -f '{{.Version}}' "$module")" = "$expected"
printf 'Verified %s@%s with GOWORK=off and no replacements\n' "$module" "$expected"
