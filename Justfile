set default-list
set lazy
set minimum-version := "1.58.0"
set shell := ["bash", "-euo", "pipefail", "-c"]

golangci_lint_version := "v2.1.6"
golangci_lint := "go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@" + golangci_lint_version
coverage_threshold := "97.0"
coverage_packages := `go list ./... | grep -vE '/(internal/testutil|provider/test/conformance)($|/)' | tr '\n' ' '`
provider_modules := "anthropic azure bedrock cohere deepinfra endpoint gemini grok huggingface ollama openai openrouter pinecone sagemaker together voyageai"
provider_packages := "./provider/anthropic/... ./provider/azure/... ./provider/bedrock/... ./provider/cohere/... ./provider/deepinfra/... ./provider/endpoint/... ./provider/gemini/... ./provider/grok/... ./provider/huggingface/... ./provider/ollama/... ./provider/openai/... ./provider/openrouter/... ./provider/pinecone/... ./provider/sagemaker/... ./provider/together/... ./provider/voyageai/..."
workspace_packages := "./... " + provider_packages + " ./harness/... ./harness/codemode/gomonty/... ./harness/sessionloop/... ./otel/... ./tui/... ./e2e/..."

# Harness module recipes
[group("modules")]
mod harness

# GoMonty adapter module recipes
[group("modules")]
mod gomonty 'harness/codemode/gomonty'

# Sessionloop module recipes
[group("modules")]
mod sessionloop 'harness/sessionloop'

# OpenTelemetry adapter module recipes
[group("modules")]
mod otel

# TUI module recipes
[group("modules")]
mod tui

# Collector-backed OpenTelemetry proof recipes
[group("modules")]
mod collector 'e2e/otel'

# Run all pure-Go workspace tests.
[group("test")]
test:
    go test -race -count=1 -timeout 60s {{ workspace_packages }}

# Run live end-to-end tests. Requires provider API keys.
[group("test")]
[working-directory("e2e")]
test-e2e:
    go test -tags=e2e -race -count=1 -timeout 300s ./...

# Run the credential-free OTLP/Collector proof.
[group("test")]
otel-e2e:
    just collector::smoke

# Measure root-module coverage, excluding test-only helpers.
[group("coverage")]
coverage:
    go test -count=1 -coverprofile=coverage.out {{ coverage_packages }}
    @go tool cover -func=coverage.out | tail -n 1

# Enforce the root module's minimum coverage.
[group("coverage")]
coverage-check: coverage
    #!/usr/bin/env bash
    set -euo pipefail
    pct="$(go tool cover -func=coverage.out | awk '/^total:/ { gsub(/%/, "", $3); print $3 }')"
    awk -v pct="$pct" -v threshold="{{ coverage_threshold }}" 'BEGIN {
      if ((pct + 0) < (threshold + 0)) {
        printf("coverage %.1f%% is below required %.1f%%\n", pct + 0, threshold + 0)
        exit 1
      }
      printf("coverage %.1f%% meets required %.1f%%\n", pct + 0, threshold + 0)
    }'

# Enforce aggregate coverage across the provider modules.
[group("coverage")]
coverage-providers:
    #!/usr/bin/env bash
    set -euo pipefail
    go test -count=1 -coverprofile=provider-coverage.out {{ provider_packages }}
    pct="$(go tool cover -func=provider-coverage.out | awk '/^total:/ { gsub(/%/, "", $3); print $3 }')"
    awk -v pct="$pct" -v threshold="{{ coverage_threshold }}" 'BEGIN {
      if ((pct + 0) < (threshold + 0)) {
        printf("provider coverage %.1f%% is below required %.1f%%\n", pct + 0, threshold + 0)
        exit 1
      }
      printf("provider coverage %.1f%% meets required %.1f%%\n", pct + 0, threshold + 0)
    }'

# Enforce every independently released pure-Go module's coverage gate.
[group("coverage")]
coverage-all: coverage-check coverage-providers
    just harness::coverage-check
    just gomonty::coverage-check
    just sessionloop::coverage-check
    just otel::coverage-check
    just tui::coverage-check

# Lint every module that does not require a native toolchain.
[group("quality")]
lint:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ golangci_lint }} run ./...
    (cd harness && {{ golangci_lint }} run ./...)
    (cd harness/codemode/gomonty && {{ golangci_lint }} run ./...)
    (cd harness/sessionloop && {{ golangci_lint }} run ./...)
    (cd otel && {{ golangci_lint }} run ./...)
    (cd tui && {{ golangci_lint }} run ./...)
    (cd e2e && {{ golangci_lint }} run --build-tags=e2e ./...)
    for module in {{ provider_modules }}; do
      echo "lint provider/$module"
      (cd "provider/$module" && {{ golangci_lint }} run ./...)
    done

# Lint the two CGO modules when their native libraries are installed.
[group("quality")]
lint-cgo:
    cd provider/local/onnx && GOWORK=off {{ golangci_lint }} run ./...
    cd e2e/localinference && GOWORK=off {{ golangci_lint }} run ./...

# Lint every module, including the native ONNX modules.
[group("quality")]
lint-all: lint lint-cgo

# Run go vet across every pure-Go workspace module.
[group("quality")]
vet:
    go vet {{ workspace_packages }}

# Build every pure-Go workspace module.
[group("quality")]
build:
    CGO_ENABLED=0 go build {{ workspace_packages }}

# Format Go source with gofmt and goimports.
[group("quality")]
fmt:
    gofmt -w .
    go run golang.org/x/tools/cmd/goimports@latest -w -local github.com/regularkevvv/agentic .

# Run formatting, vet, lint, tests, and all coverage gates.
[group("quality")]
check: fmt vet lint test coverage-all

# Remove coverage artifacts and the Go test cache.
[group("maintenance")]
clean:
    rm -f coverage.out provider-coverage.out
    go clean -testcache
