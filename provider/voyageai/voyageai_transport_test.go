package voyageai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/voyageai"
)

// singleVectorResponse is a minimal well-formed reply for one input.
const singleVectorResponse = `{
	"object":"list",
	"data":[{"object":"embedding","index":0,"embedding":[0.5,0.25]}],
	"model":"voyage-3.5",
	"usage":{"total_tokens":1}
}`

// newBodyCapturingEmbedder serves singleVectorResponse and records the decoded
// request body into *got.
func newBodyCapturingEmbedder(t *testing.T, got *map[string]any, opts ...voyageai.Option) *voyageai.Embedder {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(got); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, singleVectorResponse)
	}))
	t.Cleanup(server.Close)

	opts = append([]voyageai.Option{
		voyageai.WithAPIKey("test-key"),
		voyageai.WithBaseURL(server.URL),
		voyageai.WithHTTPClient(server.Client()),
	}, opts...)

	embedder, err := voyageai.New("voyage-3.5", opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return embedder
}

func ptr[T any](v T) *T { return &v }

// TestEmbedTruncationPrecedence pins that a per-request Truncate overrides the
// constructor default on the wire, and that omitting both leaves the field out
// so the Voyage API default applies.
func TestEmbedTruncationPrecedence(t *testing.T) {
	tests := []struct {
		name        string
		opts        []voyageai.Option
		truncate    *bool
		wantPresent bool
		want        bool
	}{
		{
			name:        "neither set omits the field",
			wantPresent: false,
		},
		{
			name:        "constructor only",
			opts:        []voyageai.Option{voyageai.WithTruncation(false)},
			wantPresent: true,
			want:        false,
		},
		{
			name:        "request only",
			truncate:    ptr(false),
			wantPresent: true,
			want:        false,
		},
		{
			name:        "request false overrides constructor true",
			opts:        []voyageai.Option{voyageai.WithTruncation(true)},
			truncate:    ptr(false),
			wantPresent: true,
			want:        false,
		},
		{
			name:        "request true overrides constructor false",
			opts:        []voyageai.Option{voyageai.WithTruncation(false)},
			truncate:    ptr(true),
			wantPresent: true,
			want:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			embedder := newBodyCapturingEmbedder(t, &gotBody, tt.opts...)

			if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
				Input:    []string{"text"},
				Truncate: tt.truncate,
			}); err != nil {
				t.Fatalf("Embed: %v", err)
			}

			got, present := gotBody["truncation"]
			if present != tt.wantPresent {
				t.Fatalf("truncation present = %v (%v), want present = %v", present, got, tt.wantPresent)
			}
			if tt.wantPresent && got != tt.want {
				t.Errorf("truncation = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestEmbedTruncateDoesNotLeakAcrossCalls pins that a per-request override
// applies to that call only and does not mutate the Embedder's default.
func TestEmbedTruncateDoesNotLeakAcrossCalls(t *testing.T) {
	var gotBody map[string]any
	embedder := newBodyCapturingEmbedder(t, &gotBody, voyageai.WithTruncation(true))

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input:    []string{"text"},
		Truncate: ptr(false),
	}); err != nil {
		t.Fatalf("Embed with override: %v", err)
	}
	if got := gotBody["truncation"]; got != false {
		t.Fatalf("overridden call truncation = %v, want false", got)
	}

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{
		Input: []string{"text"},
	}); err != nil {
		t.Fatalf("Embed without override: %v", err)
	}
	if got := gotBody["truncation"]; got != true {
		t.Errorf("subsequent call truncation = %v, want the constructor default true", got)
	}
}

// TestEmbedVectorsAreFloat32 pins the v0.3.0 element type and that values
// survive the narrower decode intact.
func TestEmbedVectorsAreFloat32(t *testing.T) {
	var gotBody map[string]any
	embedder := newBodyCapturingEmbedder(t, &gotBody)

	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"text"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	vectors := resp.Vectors
	want := [][]float32{{0.5, 0.25}}
	if len(vectors) != len(want) {
		t.Fatalf("vectors = %v, want %v", vectors, want)
	}
	for i := range want {
		if len(vectors[i]) != len(want[i]) {
			t.Fatalf("vectors[%d] = %v, want %v", i, vectors[i], want[i])
		}
		for j := range want[i] {
			if vectors[i][j] != want[i][j] {
				t.Errorf("vectors[%d][%d] = %v, want %v", i, j, vectors[i][j], want[i][j])
			}
		}
	}
}

// roundTripFunc adapts a function to http.RoundTripper so a test can inject
// transport-level failures the httptest server cannot produce.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// errReadCloser fails on Read, Close, or both.
type errReadCloser struct {
	readErr  error
	closeErr error
}

func (e errReadCloser) Read([]byte) (int, error) {
	if e.readErr != nil {
		return 0, e.readErr
	}
	return 0, io.EOF
}

func (e errReadCloser) Close() error { return e.closeErr }

func newRoundTripEmbedder(t *testing.T, fn roundTripFunc, opts ...voyageai.Option) *voyageai.Embedder {
	t.Helper()
	opts = append([]voyageai.Option{
		voyageai.WithAPIKey("test-key"),
		voyageai.WithHTTPClient(&http.Client{Transport: fn}),
	}, opts...)

	embedder, err := voyageai.New("voyage-3.5", opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return embedder
}

// TestEmbedInvalidBaseURL pins that a base URL that cannot form a request
// surfaces as an error rather than a panic.
func TestEmbedInvalidBaseURL(t *testing.T) {
	embedder, err := voyageai.New("voyage-3.5",
		voyageai.WithAPIKey("test-key"),
		voyageai.WithBaseURL("http://exa\x7fmple.com"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"hello"}}); err == nil {
		t.Fatal("Embed should fail when the base URL cannot form a request")
	}
}

// TestEmbedRetriesNetworkError pins that a transport-level failure is retried
// and that a later attempt's success is returned.
func TestEmbedRetriesNetworkError(t *testing.T) {
	var calls int
	embedder := newRoundTripEmbedder(t, func(r *http.Request) (*http.Response, error) {
		calls++
		if calls == 1 {
			return nil, errors.New("dial tcp: connection reset")
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(singleVectorResponse)),
			Header:     http.Header{"Content-Type": []string{"application/json"}},
		}, nil
	})

	resp, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
	if len(resp.Vectors) != 1 || resp.Vectors[0][0] != 0.5 {
		t.Fatalf("vectors = %v", resp.Vectors)
	}
}

// TestEmbedBodyFailures pins that a response body that cannot be read or
// closed surfaces as an error instead of being decoded as an empty payload.
func TestEmbedBodyFailures(t *testing.T) {
	tests := []struct {
		name string
		body errReadCloser
		want string
	}{
		{
			name: "read error",
			body: errReadCloser{readErr: errors.New("unexpected EOF reading body")},
			want: "unexpected EOF",
		},
		{
			name: "close error",
			body: errReadCloser{closeErr: errors.New("closing body failed")},
			want: "closing body failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			embedder := newRoundTripEmbedder(t, func(r *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body:       tt.body,
					Header:     http.Header{},
				}, nil
			})

			_, err := embedder.Embed(context.Background(), &core.EmbeddingRequest{Input: []string{"hello"}})
			if err == nil {
				t.Fatalf("Embed should fail on a %s", tt.name)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

// TestEmbedDeadlineDuringBackoff pins that a deadline expiring while the retry
// loop waits out its backoff aborts promptly with the context error, for both
// the transport-failure and the retryable-status paths. The deadline is far
// shorter than the one-second first backoff, so the wait is what trips it.
func TestEmbedDeadlineDuringBackoff(t *testing.T) {
	tests := []struct {
		name  string
		reply func() (*http.Response, error)
	}{
		{
			name:  "after a transport failure",
			reply: func() (*http.Response, error) { return nil, errors.New("dial tcp: connection reset") },
		},
		{
			name: "after a retryable status",
			reply: func() (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusTooManyRequests,
					Body:       io.NopCloser(strings.NewReader("slow down")),
					Header:     http.Header{},
				}, nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()

			var calls int
			embedder := newRoundTripEmbedder(t, func(r *http.Request) (*http.Response, error) {
				calls++
				return tt.reply()
			})

			_, err := embedder.Embed(ctx, &core.EmbeddingRequest{Input: []string{"hello"}})
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("error = %v, want context.DeadlineExceeded", err)
			}
			if calls != 1 {
				t.Errorf("calls = %d, want 1 (the expired deadline must stop the retry loop)", calls)
			}
		})
	}
}
