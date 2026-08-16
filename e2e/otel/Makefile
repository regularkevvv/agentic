.PHONY: smoke down

SMOKE_BINARY := .bin/otel-smoke

smoke:
	@set -eu; \
	architecture=$$(docker info --format '{{.Architecture}}'); \
	case "$$architecture" in \
		x86_64|amd64) goarch=amd64 ;; \
		aarch64|arm64) goarch=arm64 ;; \
		armv7l|arm) goarch=arm ;; \
		*) echo "unsupported Docker architecture: $$architecture" >&2; exit 1 ;; \
	esac; \
	mkdir -p .bin; \
	trap 'docker compose down --volumes --remove-orphans; rm -f $(SMOKE_BINARY)' EXIT; \
	docker compose down --volumes --remove-orphans; \
	GOOS=linux GOARCH="$$goarch" CGO_ENABLED=0 go -C .. test -c -o otel/$(SMOKE_BINARY) ./otel; \
	docker compose up --abort-on-container-exit --exit-code-from smoke

down:
	docker compose down --volumes --remove-orphans
