.PHONY: test lint vet build fmt check clean coverage coverage-check

GOLANGCI_LINT_VERSION := v2.1.6
COVERAGE_THRESHOLD := 97.0

test:
	go test -race -count=1 -timeout 60s ./...

coverage:
	go test -count=1 -coverprofile=coverage.out ./...
	@go tool cover -func=coverage.out | tail -n 1

coverage-check: coverage
	@pct=$$(go tool cover -func=coverage.out | awk '/^total:/ { gsub(/%/, "", $$3); print $$3 }'); \
	awk -v pct="$$pct" -v threshold="$(COVERAGE_THRESHOLD)" 'BEGIN { \
		if ((pct + 0) < (threshold + 0)) { \
			printf("coverage %.1f%% is below required %.1f%%\n", pct + 0, threshold + 0); \
			exit 1; \
		} \
		printf("coverage %.1f%% meets required %.1f%%\n", pct + 0, threshold + 0); \
	}'

lint:
	go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION) run ./...

vet:
	go vet ./...

build:
	CGO_ENABLED=0 go build ./...

fmt:
	gofmt -w .
	go run golang.org/x/tools/cmd/goimports@latest -w -local github.com/regularkevvv/agentic/harness/codemode/gomonty .

check: fmt vet lint test coverage-check

clean:
	rm -f coverage.out
	go clean -testcache
