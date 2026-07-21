package cohere_test

import (
	"net/http"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/cohere"
)

// clearAPIKeyEnv removes both key variables so a developer's real credentials
// cannot mask a test that expects the constructor to refuse.
func clearAPIKeyEnv(t *testing.T) {
	t.Helper()
	t.Setenv("CO_API_KEY", "")
	t.Setenv("COHERE_API_KEY", "")
}

// TestNewReadsAPIKeyFromEnv pins the env-var fallback and its precedence:
// CO_API_KEY is Cohere's own spelling and wins, with COHERE_API_KEY accepted.
func TestNewReadsAPIKeyFromEnv(t *testing.T) {
	tests := []struct {
		name   string
		co     string
		cohere string
		want   string
	}{
		{name: "CO_API_KEY", co: "co-key", want: "Bearer co-key"},
		{name: "COHERE_API_KEY", cohere: "cohere-key", want: "Bearer cohere-key"},
		{name: "CO_API_KEY wins", co: "co-key", cohere: "cohere-key", want: "Bearer co-key"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("CO_API_KEY", tt.co)
			t.Setenv("COHERE_API_KEY", tt.cohere)

			var gotAuth string
			embedder, err := cohere.New(cohere.DefaultEmbeddingModel,
				cohere.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					gotAuth = r.Header.Get("Authorization")
					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       http.NoBody,
						Header:     http.Header{},
					}, nil
				})}),
			)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			// The empty body fails to decode; the header is what is under test.
			_, _ = embedder.Embed(t.Context(), &core.EmbeddingRequest{Input: []string{"text"}})
			if gotAuth != tt.want {
				t.Errorf("Authorization = %q, want %q", gotAuth, tt.want)
			}
		})
	}
}

// TestNewRejectsBadConfig pins the constructor's refusals for both endpoints.
func TestNewRejectsBadConfig(t *testing.T) {
	t.Run("embedder without an API key", func(t *testing.T) {
		clearAPIKeyEnv(t)
		if _, err := cohere.New(cohere.DefaultEmbeddingModel); err == nil {
			t.Fatal("New should fail when no API key is available")
		}
	})

	t.Run("reranker without an API key", func(t *testing.T) {
		clearAPIKeyEnv(t)
		if _, err := cohere.NewReranker(cohere.DefaultRerankModel); err == nil {
			t.Fatal("NewReranker should fail when no API key is available")
		}
	})

	t.Run("empty embedder model", func(t *testing.T) {
		if _, err := cohere.New("", cohere.WithAPIKey("k")); err == nil {
			t.Fatal("New should reject an empty model")
		}
	})

	t.Run("empty reranker model", func(t *testing.T) {
		if _, err := cohere.NewReranker("", cohere.WithRerankerAPIKey("k")); err == nil {
			t.Fatal("NewReranker should reject an empty model")
		}
	})

	t.Run("negative embedder retries", func(t *testing.T) {
		if _, err := cohere.New(cohere.DefaultEmbeddingModel, cohere.WithAPIKey("k"), cohere.WithMaxRetries(-1)); err == nil {
			t.Fatal("New should reject a negative retry count")
		}
	})

	t.Run("negative reranker retries", func(t *testing.T) {
		_, err := cohere.NewReranker(cohere.DefaultRerankModel,
			cohere.WithRerankerAPIKey("k"),
			cohere.WithRerankerMaxRetries(-1),
		)
		if err == nil {
			t.Fatal("NewReranker should reject a negative retry count")
		}
	})

	t.Run("unknown default input type", func(t *testing.T) {
		_, err := cohere.New(cohere.DefaultEmbeddingModel,
			cohere.WithAPIKey("k"),
			cohere.WithDefaultInputType(cohere.InputType("image")),
		)
		if err == nil {
			t.Fatal("New should reject an input type the API does not accept")
		}
		if !strings.Contains(err.Error(), "image") {
			t.Errorf("error = %v, want it to name the rejected type", err)
		}
	})
}

// TestNewAcceptsEveryInputType pins that each documented constant is accepted
// as a default.
func TestNewAcceptsEveryInputType(t *testing.T) {
	for _, inputType := range []cohere.InputType{
		cohere.InputTypeSearchQuery,
		cohere.InputTypeSearchDocument,
		cohere.InputTypeClassification,
		cohere.InputTypeClustering,
	} {
		t.Run(string(inputType), func(t *testing.T) {
			if _, err := cohere.New(cohere.DefaultEmbeddingModel,
				cohere.WithAPIKey("k"),
				cohere.WithDefaultInputType(inputType),
			); err != nil {
				t.Errorf("New with %q: %v", inputType, err)
			}
		})
	}
}

// TestName pins that both types report the configured model.
func TestName(t *testing.T) {
	embedder, err := cohere.New("embed-multilingual-v3.0", cohere.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := embedder.Name(); got != "embed-multilingual-v3.0" {
		t.Errorf("Embedder.Name() = %q, want the configured model", got)
	}

	reranker, err := cohere.NewReranker("rerank-english-v3.0", cohere.WithRerankerAPIKey("k"))
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	if got := reranker.Name(); got != "rerank-english-v3.0" {
		t.Errorf("Reranker.Name() = %q, want the configured model", got)
	}
}

// TestMustNew pins that the Must variants return a usable value on success and
// panic on the failure the plain constructor reports.
func TestMustNew(t *testing.T) {
	t.Run("embedder success", func(t *testing.T) {
		if got := cohere.MustNew(cohere.DefaultEmbeddingModel, cohere.WithAPIKey("k")).Name(); got != cohere.DefaultEmbeddingModel {
			t.Errorf("Name() = %q, want %q", got, cohere.DefaultEmbeddingModel)
		}
	})

	t.Run("reranker success", func(t *testing.T) {
		if got := cohere.MustNewReranker(cohere.DefaultRerankModel, cohere.WithRerankerAPIKey("k")).Name(); got != cohere.DefaultRerankModel {
			t.Errorf("Name() = %q, want %q", got, cohere.DefaultRerankModel)
		}
	})

	t.Run("embedder panics", func(t *testing.T) {
		clearAPIKeyEnv(t)
		defer func() {
			if recover() == nil {
				t.Error("MustNew should panic when New would fail")
			}
		}()
		_ = cohere.MustNew(cohere.DefaultEmbeddingModel)
	})

	t.Run("reranker panics", func(t *testing.T) {
		clearAPIKeyEnv(t)
		defer func() {
			if recover() == nil {
				t.Error("MustNewReranker should panic when NewReranker would fail")
			}
		}()
		_ = cohere.MustNewReranker(cohere.DefaultRerankModel)
	})
}

// TestInterfaceSatisfaction pins that the exported types are usable through the
// core abstractions, which is the only way callers reach them.
func TestInterfaceSatisfaction(t *testing.T) {
	var _ core.Embedder = cohere.MustNew(cohere.DefaultEmbeddingModel, cohere.WithAPIKey("k"))
	var _ core.Reranker = cohere.MustNewReranker(cohere.DefaultRerankModel, cohere.WithRerankerAPIKey("k"))
}
