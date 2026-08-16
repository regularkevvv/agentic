.PHONY: test lint lint-all lint-cgo vet build fmt check clean test-e2e coverage coverage-check coverage-all

GOLANGCI_LINT_VERSION := v2.1.6
GOLANGCI_LINT := go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)
COVERAGE_PACKAGES := $(shell go list ./... | grep -vE '/(internal/testutil|provider/test/conformance)($$|/)')
COVERAGE_THRESHOLD := 97.0
WORKSPACE_PACKAGES := ./... ./harness/... ./harness/codemode/gomonty/... ./harness/sessionloop/... ./otel/... ./tui/... ./e2e/...

# Run all tests
test:
	go test -race -count=1 -timeout 60s $(WORKSPACE_PACKAGES)

# Run end-to-end tests (requires API keys). They are their own module, so they
# are invisible to every other target here.
test-e2e:
	cd e2e && go test -tags=e2e -race -count=1 -timeout 300s ./...

# Run coverage excluding test-only helper packages
coverage:
	go test -count=1 -coverprofile=coverage.out $(COVERAGE_PACKAGES)
	@go tool cover -func=coverage.out | tail -n 1

# Enforce minimum coverage threshold for CI
coverage-check: coverage
	@pct=$$(go tool cover -func=coverage.out | awk '/^total:/ { gsub(/%/, "", $$3); print $$3 }'); \
	awk -v pct="$$pct" -v threshold="$(COVERAGE_THRESHOLD)" 'BEGIN { \
		if ((pct + 0) < (threshold + 0)) { \
			printf("coverage %.1f%% is below required %.1f%%\n", pct + 0, threshold + 0); \
			exit 1; \
		} \
		printf("coverage %.1f%% meets required %.1f%%\n", pct + 0, threshold + 0); \
	}'

# Each released pure-Go production module owns an independent 97% gate. E2E and
# test hosts prove integration behavior and are intentionally not coverage
# padding. The native ONNX module gates its CGO build and recorded real-model
# compatibility separately because its inference paths require an external
# runtime, tokenizer, and exported graph.
coverage-all: coverage-check
	$(MAKE) -C harness coverage-check
	$(MAKE) -C harness/codemode/gomonty coverage-check
	$(MAKE) -C harness/sessionloop coverage-check
	$(MAKE) -C otel coverage-check
	$(MAKE) -C tui coverage-check

# Lint every module that builds without a native toolchain (same version as CI).
#
# The e2e run passes -e2e explicitly: the live tests are the only files in
# e2e/providers, so without the tag golangci-lint finds no Go files there and
# reports a clean result over a package it never read.
lint:
	$(GOLANGCI_LINT) run ./...
	cd harness && $(GOLANGCI_LINT) run ./...
	cd harness/codemode/gomonty && $(GOLANGCI_LINT) run ./...
	cd harness/sessionloop && $(GOLANGCI_LINT) run ./...
	cd otel && $(GOLANGCI_LINT) run ./...
	cd tui && $(GOLANGCI_LINT) run ./...
	cd e2e && $(GOLANGCI_LINT) run --build-tags=e2e ./...

# The two CGO modules are linted apart from the others, not forgotten by them.
# Linting compiles, and compiling these means linking cgo against ONNX Runtime
# and libtokenizers.a; requiring those for `make lint` would make the ordinary
# contributor loop depend on the one thing this repository deliberately keeps
# optional. CI runs them as its own job, so they are gated, just not here.
# See provider/local/onnx/README.md for the two downloads.
lint-cgo:
	cd provider/local/onnx && GOWORK=off $(GOLANGCI_LINT) run ./...
	cd e2e/localinference && GOWORK=off $(GOLANGCI_LINT) run ./...

# Every module, for when the native libraries are installed.
lint-all: lint lint-cgo

# Run go vet
vet:
	go vet $(WORKSPACE_PACKAGES)

# Build all packages
build:
	CGO_ENABLED=0 go build $(WORKSPACE_PACKAGES)

# Format code
fmt:
	gofmt -w .
	go run golang.org/x/tools/cmd/goimports@latest -w -local github.com/regularkevvv/agentic .

# Run all checks (test + lint + vet)
check: fmt vet lint test coverage-all

# Remove build artifacts
clean:
	rm -f coverage.out
	go clean -testcache
