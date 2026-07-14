.PHONY: test lint vet build fmt check clean test-e2e coverage coverage-check

GOLANGCI_LINT_VERSION := v2.1.6
COVERAGE_PACKAGES := $(shell go list ./... | grep -vE '/(e2e|examples|internal/testutil)($$|/)')
COVERAGE_THRESHOLD := 97.0

# Run all tests (excluding e2e)
test:
	go test -race -count=1 -timeout 60s $$(go list ./... | grep -v /e2e)

# Run end-to-end tests (requires API keys)
test-e2e:
	go test -tags=e2e -race -count=1 -timeout 300s ./e2e/...

# Run coverage excluding e2e, examples, and test-only helper packages
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

# Run linter (same version as CI)
lint:
	go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION) run ./...

# Run go vet
vet:
	go vet ./...

# Build all packages
build:
	go build ./...

# Format code
fmt:
	gofmt -w .
	go run golang.org/x/tools/cmd/goimports@latest -w -local github.com/regularkevvv/agentic .

# Run all checks (test + lint + vet)
check: fmt vet lint test

# Remove build artifacts
clean:
	rm -f coverage.out
	go clean -testcache
