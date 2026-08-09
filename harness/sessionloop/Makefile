.PHONY: test lint vet build fmt check clean coverage coverage-check

# Module gates verify the standalone module graph (plan section 12.4); they
# must pass whether or not the repository workspace lists this module.
export GOWORK := off

GOLANGCI_LINT_VERSION := v2.1.6
COVERAGE_THRESHOLD := 97.0
# conformance/ and testkit/ are reusable test infrastructure, excluded from
# the gate like the harness storetest helpers; both are still exercised by
# the main package's tests, which run the conformance suite against the
# testkit host.
COVERAGE_PACKAGES := $(shell GOWORK=off go list ./... | grep -Ev '/(conformance|testkit)($$|/)')

test:
	go test -race -count=1 -timeout 60s ./...

coverage:
	go test -count=1 -coverprofile=coverage.out $(COVERAGE_PACKAGES)
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
	go run golang.org/x/tools/cmd/goimports@latest -w -local github.com/regularkevvv/agentic/harness/sessionloop .

check: fmt vet lint test coverage-check

clean:
	rm -f coverage.out
	go clean -testcache
