.PHONY: test lint vet build fmt check coverage coverage-check

GOLANGCI_LINT_VERSION := v2.1.6
COVERAGE_THRESHOLD := 97.0

test:
	go test -race -count=1 -timeout 60s ./...

coverage:
	go test -count=1 -coverprofile=coverage.out ./...
	@go tool cover -func=coverage.out | tail -n 1

coverage-check: coverage
	@pct=$$(go tool cover -func=coverage.out | awk '/^total:/ { gsub(/%/, "", $$3); print $$3 }'); \
	awk -v pct="$$pct" -v threshold="$(COVERAGE_THRESHOLD)" 'BEGIN { if ((pct + 0) < (threshold + 0)) exit 1 }'

lint:
	go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION) run ./...

vet:
	go vet ./...

build:
	go build ./...

fmt:
	gofmt -w .

check: fmt vet test coverage-check
