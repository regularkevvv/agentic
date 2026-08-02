# Contributing to Agentic

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/agentic.git
   cd agentic
   ```
3. Create a branch for your work:
   ```bash
   git checkout -b feature/my-change
   ```

## Development

### Prerequisites

- Go 1.25.4 or later
- [golangci-lint](https://golangci-lint.run/welcome/install/) (for linting)

### Common Commands

```bash
make test      # run tests
make coverage  # run coverage check inputs (root module only)
make lint      # run linter
make vet       # run go vet
make fmt       # format code
make check     # run all checks (fmt + vet + lint + test)
make build     # build all packages
```

### Running E2E Tests

E2E tests require API keys configured in `.env`:

```bash
cp .env.example .env   # fill in your keys
make test-e2e
```

## Submitting Changes

1. Run `make check` and ensure it passes
2. Commit with a clear message
3. Push to your fork and open a pull request against `develop`

### Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if you change public APIs
- Write a clear description explaining **what** and **why**

### Test Organization

- Keep tests close to the subject they verify; avoid catch-all files like `coverage_*_test.go`
- Prefer `*_test.go` for behavior tests, `*_internal_test.go` for same-package white-box tests, and `*_transport_test.go` for local protocol/server tests
- Reserve `e2e/providers/` for smoke tests that hit real external APIs and run them via `make test-e2e`. `e2e/` is its own module, so anything it imports stays out of the library's dependency graph — and it may import only the root module's public packages
- `provider/local/onnx/` is its own module too, is not in `go.work`, and tests itself. None of the commands above reach it, deliberately: it needs CGO, a native ONNX Runtime, and a statically linked tokenizer. See [`provider/local/onnx/README.md`](provider/local/onnx/README.md) before `cd provider/local/onnx`

## Reporting Bugs

Open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Go version and OS

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
