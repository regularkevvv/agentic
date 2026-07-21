package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/bedrock"
)

// embedServer stands in for the Bedrock Runtime InvokeModel endpoint. It
// records every request body and replies with respond(body), letting the
// handler vary its answer by input, and stamps the input-token-count header
// that carries usage for embedding models.
type embedServer struct {
	mu     sync.Mutex
	bodies []map[string]any

	// tokensPerCall is reported in the x-amzn-bedrock-input-token-count
	// header on every reply.
	tokensPerCall string
}

// start brings up an httptest server and returns a Bedrock Runtime client
// pointed at it, so assertions run against the bytes the AWS SDK actually puts
// on the wire and against a genuine HTTP response carrying real headers.
func (s *embedServer) start(t *testing.T, respond func(body map[string]any) string) *bedrockruntime.Client {
	t.Helper()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode request body: %v", err)
			return
		}

		s.mu.Lock()
		s.bodies = append(s.bodies, body)
		s.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if s.tokensPerCall != "" {
			w.Header().Set("x-amzn-bedrock-input-token-count", s.tokensPerCall)
		}
		_, _ = w.Write([]byte(respond(body)))
	}))
	t.Cleanup(srv.Close)

	return bedrockruntime.New(bedrockruntime.Options{
		Region:       "us-east-1",
		BaseEndpoint: aws.String(srv.URL),
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
	})
}

// calls returns the request bodies captured so far.
func (s *embedServer) calls() []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bodies
}

// Usage for embedding models arrives in a response header rather than the
// response body, and Titan fans one call out per input, so the counts have to
// be summed across the whole fan-out. This exercises the real SDK middleware
// stack that preserves the raw HTTP response.
func TestTransportTitanSumsUsageFromResponseHeader(t *testing.T) {
	srv := &embedServer{tokensPerCall: "7"}
	client := srv.start(t, func(body map[string]any) string {
		// Titan echoes one vector per call; vary it by input so a
		// reordered fan-out would be visible.
		if body["inputText"] == "second" {
			return `{"embedding": [0.2], "inputTextTokenCount": 99}`
		}
		return `{"embedding": [0.1], "inputTextTokenCount": 99}`
	})

	embedder, err := bedrock.NewEmbedder("amazon.titan-embed-text-v2:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input: []string{"first", "second"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	// Two calls at 7 tokens each. The body's inputTextTokenCount is only a
	// fallback and must not win over a header that was present.
	if resp.Usage.PromptTokens != 14 || resp.Usage.TotalTokens != 14 {
		t.Errorf("expected 14 tokens summed from the header, got %+v", resp.Usage)
	}
	if len(resp.Vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(resp.Vectors))
	}
	if resp.Vectors[0][0] != 0.1 || resp.Vectors[1][0] != 0.2 {
		t.Errorf("vectors came back out of input order: %v", resp.Vectors)
	}
	if resp.Model != "amazon.titan-embed-text-v2:0" {
		t.Errorf("unexpected model %q", resp.Model)
	}
	if got := len(srv.calls()); got != 2 {
		t.Errorf("expected one call per input, got %d", got)
	}
}

// A response whose header a proxy stripped must fall back to the count Titan
// also reports in its body rather than reporting nothing.
func TestTransportTitanFallsBackToBodyTokenCount(t *testing.T) {
	srv := &embedServer{}
	client := srv.start(t, func(map[string]any) string {
		return `{"embedding": [0.1], "inputTextTokenCount": 11}`
	})

	embedder, err := bedrock.NewEmbedder("amazon.titan-embed-text-v2:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Usage.PromptTokens != 11 {
		t.Errorf("expected the body token count as fallback, got %+v", resp.Usage)
	}
}

// Cohere on Bedrock takes the whole batch in one call and requires input_type,
// which the SDK must serialize under the model's own field names.
func TestTransportCohereSendsBatchAndReadsHeaderUsage(t *testing.T) {
	srv := &embedServer{tokensPerCall: "23"}
	client := srv.start(t, func(map[string]any) string {
		return `{"embeddings": [[0.1], [0.2]]}`
	})

	embedder, err := bedrock.NewEmbedder("cohere.embed-english-v3", bedrock.WithClient(client))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input:     []string{"first", "second"},
		InputType: core.EmbeddingInputQuery,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	calls := srv.calls()
	if len(calls) != 1 {
		t.Fatalf("expected the batch to go out in one call, got %d", len(calls))
	}
	texts, ok := calls[0]["texts"].([]any)
	if !ok || len(texts) != 2 {
		t.Fatalf("expected 2 texts on the wire, got %v", calls[0]["texts"])
	}
	if calls[0]["input_type"] != "search_query" {
		t.Errorf("expected input_type search_query, got %v", calls[0]["input_type"])
	}
	if resp.Usage.PromptTokens != 23 {
		t.Errorf("expected 23 tokens from the header, got %+v", resp.Usage)
	}
	if len(resp.Vectors) != 2 {
		t.Errorf("expected 2 vectors, got %d", len(resp.Vectors))
	}
}

// An error status from Bedrock must surface as an error rather than as an
// empty result.
func TestTransportEmbedPropagatesAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message": "malformed input"}`))
	}))
	t.Cleanup(srv.Close)

	client := bedrockruntime.New(bedrockruntime.Options{
		Region:           "us-east-1",
		BaseEndpoint:     aws.String(srv.URL),
		Credentials:      credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
		RetryMaxAttempts: 1,
	})

	embedder, err := bedrock.NewEmbedder("amazon.titan-embed-text-v2:0", bedrock.WithClient(client))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("expected an error status to be reported")
	}
}
