package gemini

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNewEmbedderDefaultsToGeminiEmbedding001(t *testing.T) {
	e, err := NewEmbedder("", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	if e.Name() != DefaultEmbeddingModel {
		t.Errorf("expected %q, got %q", DefaultEmbeddingModel, e.Name())
	}
}

func TestNewEmbedderKeepsExplicitModel(t *testing.T) {
	e, err := NewEmbedder("text-embedding-005", WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	if e.Name() != "text-embedding-005" {
		t.Errorf("expected %q, got %q", "text-embedding-005", e.Name())
	}
}

// The gemini-embedding-2 family conditions on a text prefix rather than the
// taskType field this Embedder sets, so a RETRIEVAL_* mapping would be
// accepted and ignored, degrading retrieval with no error. Construction must
// refuse the whole family, including the -preview variant that the upstream
// reference's exact-string check lets through.
func TestNewEmbedderRejectsTaskPrefixModels(t *testing.T) {
	for _, model := range []string{
		"gemini-embedding-2",
		"gemini-embedding-2-preview",
		"gemini-embedding-2-exp-01-01",
	} {
		t.Run(model, func(t *testing.T) {
			_, err := NewEmbedder(model, WithAPIKey("test-key"))
			if err == nil {
				t.Fatalf("expected an error for %q", model)
			}
			if !strings.Contains(err.Error(), "text prefix") {
				t.Errorf("expected a prefix-conditioning explanation, got %v", err)
			}
		})
	}
}

func TestNewEmbedderNoAPIKey(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")

	if _, err := NewEmbedder("gemini-embedding-001"); err == nil {
		t.Error("expected an error when no API key is set")
	}
}

func TestNewEmbedderVertexAI(t *testing.T) {
	// A custom base URL plus an EMPTY project and location keeps Vertex client
	// creation from reaching for application default credentials, matching
	// TestGeminiNewVertexAIWithCustomBaseURL.
	//
	// Both halves matter. Naming a project sends the SDK down the full Vertex
	// auth path, which resolves ADC and fails anywhere they are not
	// configured — so a test that names one passes on a developer machine
	// running gcloud and fails in CI. WithVertexAI's project and location
	// plumbing is covered without a client by TestWithVertexAIOption.
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()
	t.Setenv("GOOGLE_VERTEX_BASE_URL", server.URL)

	e, err := NewEmbedder("gemini-embedding-001", WithVertexAI("", ""))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	if !e.vertexAI {
		t.Error("expected the Vertex AI backend to be recorded")
	}
}

func TestMustNewEmbedder(t *testing.T) {
	if e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key")); e.Name() != "gemini-embedding-001" {
		t.Errorf("unexpected name %q", e.Name())
	}
}

func TestMustNewEmbedderPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected a panic for a rejected model")
		}
	}()
	MustNewEmbedder("gemini-embedding-2", WithAPIKey("test-key"))
}

func TestIsTaskPrefixModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		{"gemini-embedding-2", true},
		{"gemini-embedding-2-preview", true},
		{"gemini-embedding-001", false},
		{"text-embedding-005", false},
		{"gemini-embedding-20", true},
	}
	for _, tt := range tests {
		if got := isTaskPrefixModel(tt.model); got != tt.want {
			t.Errorf("isTaskPrefixModel(%q) = %v, want %v", tt.model, got, tt.want)
		}
	}
}

func TestResolveTaskType(t *testing.T) {
	tests := []struct {
		name       string
		configured string
		inputType  core.EmbeddingInputType
		want       string
	}{
		{"query maps to retrieval query", "", core.EmbeddingInputQuery, TaskTypeRetrievalQuery},
		{"document maps to retrieval document", "", core.EmbeddingInputDocument, TaskTypeRetrievalDocument},
		{"none leaves task type unset", "", core.EmbeddingInputNone, ""},
		{"none falls back to the configured task", TaskTypeClustering, core.EmbeddingInputNone, TaskTypeClustering},
		// A per-request setting beats a constructor default, matching the
		// precedence the other providers in this tree use.
		{"query overrides the configured task", TaskTypeClustering, core.EmbeddingInputQuery, TaskTypeRetrievalQuery},
		{"document overrides the configured task", TaskTypeClassification, core.EmbeddingInputDocument, TaskTypeRetrievalDocument},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := &Embedder{taskType: tt.configured}
			if got := e.resolveTaskType(tt.inputType); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestWithEmbeddingTaskType(t *testing.T) {
	e, err := NewEmbedder("gemini-embedding-001", WithAPIKey("test-key"), WithEmbeddingTaskType(TaskTypeFactVerification))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	if e.taskType != TaskTypeFactVerification {
		t.Errorf("got %q, want %q", e.taskType, TaskTypeFactVerification)
	}
}

func TestNewEmbedderSingleInput(t *testing.T) {
	tests := []struct {
		name     string
		vertexAI bool
		model    string
		want     bool
	}{
		{"gemini api batches natively", false, "gemini-embedding-exp", false},
		{"vertex exempts gemini-embedding-001", true, "gemini-embedding-001", false},
		{"vertex caps other gemini models", true, "gemini-embedding-exp-03-07", true},
		{"vertex caps maas models", true, "some-maas-model", true},
		{"vertex batches non-gemini models", true, "text-embedding-005", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := newEmbedderSingleInput(tt.vertexAI, tt.model); got != tt.want {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBuildConfigDimensions(t *testing.T) {
	e := &Embedder{}

	cfg, err := e.buildConfig(&core.EmbeddingRequest{Input: []string{"a"}})
	if err != nil {
		t.Fatalf("buildConfig: %v", err)
	}
	if cfg.OutputDimensionality != nil {
		t.Error("expected no dimensionality override when Dimensions is zero")
	}

	cfg, err = e.buildConfig(&core.EmbeddingRequest{Input: []string{"a"}, Dimensions: 768})
	if err != nil {
		t.Fatalf("buildConfig: %v", err)
	}
	if cfg.OutputDimensionality == nil || *cfg.OutputDimensionality != 768 {
		t.Errorf("expected 768, got %v", cfg.OutputDimensionality)
	}
}

// Truncate=false asks the provider to reject over-length input. The SDK tags
// AutoTruncate omitempty, so false never reaches the wire and Vertex applies
// its own truncating default. Honoring the request is impossible, so it must
// fail rather than truncate silently.
func TestBuildConfigRejectsTruncateFalse(t *testing.T) {
	no := false
	for _, e := range []*Embedder{{}, {vertexAI: true}} {
		_, err := e.buildConfig(&core.EmbeddingRequest{Input: []string{"a"}, Truncate: &no})
		if err == nil {
			t.Fatal("expected an error for Truncate=false")
		}
		if !strings.Contains(err.Error(), "not supported") {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func TestBuildConfigTruncateTrueNeedsVertex(t *testing.T) {
	yes := true

	e := &Embedder{}
	if _, err := e.buildConfig(&core.EmbeddingRequest{Input: []string{"a"}, Truncate: &yes}); err == nil {
		t.Error("expected an error for Truncate=true on the Gemini API")
	}

	vertex := &Embedder{vertexAI: true}
	cfg, err := vertex.buildConfig(&core.EmbeddingRequest{Input: []string{"a"}, Truncate: &yes})
	if err != nil {
		t.Fatalf("buildConfig: %v", err)
	}
	if !cfg.AutoTruncate {
		t.Error("expected AutoTruncate to be set on Vertex")
	}
}

func TestBuildConfigTruncateNilOmitsIt(t *testing.T) {
	e := &Embedder{vertexAI: true}
	cfg, err := e.buildConfig(&core.EmbeddingRequest{Input: []string{"a"}})
	if err != nil {
		t.Fatalf("buildConfig: %v", err)
	}
	if cfg.AutoTruncate {
		t.Error("expected AutoTruncate to stay unset when Truncate is nil")
	}
}

func TestEmbedValidatesRequest(t *testing.T) {
	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))

	tests := []struct {
		name string
		req  *core.EmbeddingRequest
	}{
		{"empty input", &core.EmbeddingRequest{}},
		{"empty string input", &core.EmbeddingRequest{Input: []string{""}}},
		{"bad input type", &core.EmbeddingRequest{Input: []string{"a"}, InputType: "nonsense"}},
		{"negative dimensions", &core.EmbeddingRequest{Input: []string{"a"}, Dimensions: -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := e.Embed(context.Background(), tt.req); err == nil {
				t.Error("expected a validation error")
			}
		})
	}
}

// Embed must reject an unrepresentable Truncate before issuing any request.
func TestEmbedRejectsUnsupportedTruncateBeforeCalling(t *testing.T) {
	e := MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
	no := false
	if _, err := e.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"a"}, Truncate: &no}); err == nil {
		t.Error("expected an error for Truncate=false")
	}
}

func TestEmbedderImplementsCoreEmbedder(t *testing.T) {
	var _ core.Embedder = MustNewEmbedder("gemini-embedding-001", WithAPIKey("test-key"))
}
