package agentic_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// consumerProgram is what a downstream retrieval system does with this API: it
// implements the encoder interface, drives a provider through the helpers, and
// persists space identity beside the values.
//
// It imports only the root package. That is the point of the test — every type
// it names must be reachable from the facade, with no internal package in
// sight and no workspace to fall back on.
const consumerProgram = `package main

import (
	"context"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/deepinfra"
	testprovider "github.com/regularkevvv/agentic/provider/test"
)

// storeEncoder is a consumer's own encoder, written against the public
// interface alone.
type storeEncoder struct {
	inner agentic.RepresentationEncoder
}

func (e *storeEncoder) Encode(ctx context.Context, req *agentic.RepresentationRequest) (*agentic.RepresentationResponse, error) {
	return e.inner.Encode(ctx, req)
}

func (e *storeEncoder) Name() string                                   { return e.inner.Name() }
func (e *storeEncoder) Capabilities() agentic.RepresentationCapabilities { return e.inner.Capabilities() }

var _ agentic.RepresentationEncoder = (*storeEncoder)(nil)

// indexedDocument is what the consumer persists: the values, and the identity
// of the spaces they can be compared within.
type indexedDocument struct {
	Dense        []float32
	Sparse       *agentic.SparseVector
	DenseSpaceID string
	SparseSpace  agentic.VectorSpace
}

func main() {
	ctx := context.Background()
	encoder := &storeEncoder{inner: testprovider.NewTestRepresentationEncoder()}

	resp, err := agentic.EncodeDocuments(ctx, encoder,
		[]string{"PostgreSQL supports sparse vectors"},
		agentic.RepresentationDense,
		agentic.RepresentationSparse,
	)
	if err != nil {
		panic(err)
	}

	denseSpace, ok := resp.Space(agentic.RepresentationDense)
	if !ok {
		panic("no dense space")
	}
	sparseSpace := resp.Spaces[agentic.RepresentationSparse]
	if sparseSpace.Metric != agentic.SimilarityDotProduct {
		panic("unexpected sparse metric")
	}
	if err := denseSpace.Validate(); err != nil {
		panic(err)
	}
	if denseSpace.ID != denseSpace.CanonicalID() {
		panic("space ID is not reproducible")
	}
	if denseSpace.Compatible(sparseSpace) {
		panic("two kinds must not share a space")
	}

	doc := indexedDocument{
		Dense:        resp.Data[0].Dense,
		Sparse:       resp.Data[0].Sparse,
		DenseSpaceID: denseSpace.ID,
		SparseSpace:  sparseSpace,
	}
	if len(doc.Dense) != denseSpace.Dimensions || doc.Sparse.Len() == 0 {
		panic("unexpected representation shape")
	}

	// Typed errors are branchable without naming a concrete type.
	_, err = agentic.EncodeQueries(ctx, encoder, []string{"a"}, agentic.RepresentationKind("colbert"))
	if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
		panic("expected an invalid request error")
	}

	// Both adapters are reachable from the facade.
	adapted, err := agentic.EmbedderAsRepresentationEncoder(
		testprovider.NewTestEmbedder(4),
		agentic.VectorSpace{Provider: "test", Revision: "2026-01"},
	)
	if err != nil {
		panic(err)
	}
	if _, err := agentic.EncodeDocuments(ctx, adapted, []string{"a"}, agentic.RepresentationDense); err != nil {
		panic(err)
	}
	if _, err := agentic.RepresentationEncoderAsEmbedder(encoder); err != nil {
		panic(err)
	}

	// A real provider satisfies both contracts without the consumer importing
	// anything internal.
	live := deepinfra.MustNew(deepinfra.BGEM3Model, deepinfra.WithAPIToken("unused-in-this-check"))
	var _ agentic.RepresentationEncoder = live
	var _ agentic.Embedder = live
	if !live.Capabilities().Supports(agentic.RepresentationSparse) {
		panic("expected sparse support")
	}

	// A validator built from the facade enforces the same contract.
	validator := agentic.RepresentationValidator{
		Provider:     "consumer",
		Capabilities: agentic.RepresentationCapabilities{Outputs: []agentic.RepresentationKind{agentic.RepresentationDense}},
		Limits:       agentic.RepresentationLimits{MaxInputs: 1},
	}
	err = validator.ValidateRequest(&agentic.RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []agentic.RepresentationKind{agentic.RepresentationDense},
	})
	if !errors.Is(err, agentic.ErrInvalidRepresentationRequest) {
		panic("expected the configured limit to be enforced")
	}

	fmt.Println("ok")
}
`

// TestFreshConsumerUsesOnlyThePublicAPI builds and runs a throwaway module
// that consumes this one from outside the workspace.
//
// It catches the failure the in-repo tests structurally cannot: a facade that
// compiles here only because the workspace or an internal import is in scope.
// A downstream retrieval system has neither, so the check runs with GOWORK=off
// against a module that names nothing but the public API.
//
// The replace directive stands in for a published tag. What is being proven is
// self-sufficiency, not distribution — whether the tag resolves is the release
// process's problem, whether the API needs internals is this test's.
func TestFreshConsumerUsesOnlyThePublicAPI(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping fresh-consumer build in short mode")
	}

	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Dir(currentFile)

	dir := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	goVersion, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}

	write("go.mod", strings.Join([]string{
		"module agenticconsumer",
		"",
		goDirective(string(goVersion)),
		"",
		"require github.com/regularkevvv/agentic v0.0.0",
		"",
		"replace github.com/regularkevvv/agentic => " + root,
		"",
	}, "\n"))
	write("main.go", consumerProgram)

	// The consumer inherits this module's checksums, so its transitive
	// requirements resolve from the module cache without reaching a proxy.
	sums, err := os.ReadFile(filepath.Join(root, "go.sum"))
	if err != nil {
		t.Fatalf("read go.sum: %v", err)
	}
	write("go.sum", string(sums))

	command := exec.Command("go", "run", ".")
	command.Dir = dir
	command.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod")

	output, err := command.CombinedOutput()
	if err != nil {
		if isNetworkFailure(string(output)) {
			t.Skipf("module resolution needs a proxy that is not reachable:\n%s", output)
		}
		t.Fatalf("fresh consumer failed:\n%s", output)
	}
	if !strings.Contains(string(output), "ok") {
		t.Fatalf("fresh consumer produced unexpected output:\n%s", output)
	}
}

// goDirective extracts the go language line from a go.mod, so the consumer
// module targets the same version rather than a hard-coded one that drifts.
func goDirective(gomod string) string {
	for _, line := range strings.Split(gomod, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "go ") {
			return strings.TrimSpace(line)
		}
	}
	return "go 1.25"
}

// isNetworkFailure reports whether the go tool failed to reach a proxy rather
// than failing to compile. Skipping only on this narrow condition keeps the
// gate meaningful when the network is available, which is where it runs.
func isNetworkFailure(output string) bool {
	for _, marker := range []string{
		"dial tcp",
		"proxy.golang.org",
		"no such host",
		"connection refused",
		"i/o timeout",
	} {
		if strings.Contains(output, marker) {
			return true
		}
	}
	return false
}
