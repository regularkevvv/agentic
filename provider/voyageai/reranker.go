package voyageai

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"

	"github.com/regularkevvv/agentic/internal/core"
)

// Reranker implements core.Reranker using the Voyage AI rerank API.
//
// A reranker is a cross-encoder: it reads the query and each document together
// rather than comparing independently computed vectors, which is far more
// accurate and far more expensive than embedding similarity. Pair it with an
// Embedder — retrieve a shortlist by vector search, then reorder that
// shortlist here.
type Reranker struct {
	*client

	model      string
	truncation *bool
}

// RerankerOption configures the Voyage AI Reranker.
//
// This is deliberately a distinct type from Option even though the transport
// knobs overlap: the two endpoints take different model families and different
// per-call fields, so the compiler, not a runtime error, is what stops an
// embedding option from being passed to NewReranker or the reverse.
type RerankerOption func(*rerankerConfig)

type rerankerConfig struct {
	transportConfig

	truncation *bool
}

// WithRerankerAPIKey sets the API key. If not set, the VOYAGE_API_KEY env var
// is used.
func WithRerankerAPIKey(apiKey string) RerankerOption {
	return func(c *rerankerConfig) { c.apiKey = apiKey }
}

// WithRerankerBaseURL sets a custom base URL (default
// https://api.voyageai.com/v1).
func WithRerankerBaseURL(baseURL string) RerankerOption {
	return func(c *rerankerConfig) { c.baseURL = baseURL }
}

// WithRerankerHTTPClient sets a custom HTTP client.
func WithRerankerHTTPClient(client *http.Client) RerankerOption {
	return func(c *rerankerConfig) { c.httpClient = client }
}

// WithRerankerMaxRetries sets how many times a request is retried on 429 and
// 5xx responses (default 2, matching the Embedder).
func WithRerankerMaxRetries(retries int) RerankerOption {
	return func(c *rerankerConfig) { c.maxRetries = &retries }
}

// WithRerankerTruncation sets what happens to a query/document pair longer
// than the model's context length: true truncates it, false rejects the whole
// request with an error.
//
// Unlike the Embedder's WithTruncation this has no per-call override, because
// core.RerankRequest carries no truncation field. If it is not set the Voyage
// API's own default applies, which is to truncate. Scoring a clipped document
// is less costly than indexing one — the mistake is discarded after the call
// rather than stored — but it still silently ranks on partial text.
func WithRerankerTruncation(truncation bool) RerankerOption {
	return func(c *rerankerConfig) { c.truncation = &truncation }
}

// NewReranker creates a new Voyage AI Reranker.
//
// Examples:
//
//	reranker, err := voyageai.NewReranker("rerank-2.5", voyageai.WithRerankerAPIKey("pa-..."))
//	reranker, err := voyageai.NewReranker("rerank-2.5-lite") // key from VOYAGE_API_KEY
func NewReranker(model string, opts ...RerankerOption) (*Reranker, error) {
	cfg := &rerankerConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	c, err := newClient(cfg.transportConfig)
	if err != nil {
		return nil, err
	}

	return &Reranker{
		client:     c,
		model:      model,
		truncation: cfg.truncation,
	}, nil
}

// MustNewReranker is like NewReranker but panics on error.
func MustNewReranker(model string, opts ...RerankerOption) *Reranker {
	r, err := NewReranker(model, opts...)
	if err != nil {
		panic(err)
	}
	return r
}

// rerankRequest is the Voyage AI rerank wire request.
//
// Note the field name: Voyage calls the result cap top_k, not the top_n used
// by core.RerankRequest and by Cohere. This is verified against the live API —
// sending top_n is rejected outright with "Argument 'top_n' is not supported by
// our API" — so do not "correct" it to match the core field or Cohere.
//
// return_documents is deliberately never sent: echoing every document back
// doubles the response size to restate what the caller already holds, and
// core.RerankResult.Document is filled from the request slice regardless.
type rerankRequest struct {
	Query      string   `json:"query"`
	Documents  []string `json:"documents"`
	Model      string   `json:"model"`
	TopK       int      `json:"top_k,omitempty"`
	Truncation *bool    `json:"truncation,omitempty"`
}

type rerankResponse struct {
	Data []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Rerank implements core.Reranker.
func (r *Reranker) Rerank(ctx context.Context, req *core.RerankRequest) (*core.RerankResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	body := rerankRequest{
		Query:      req.Query,
		Documents:  req.Documents,
		Model:      r.model,
		TopK:       min(req.TopN, len(req.Documents)),
		Truncation: r.truncation,
	}

	payload, err := r.post(ctx, "/rerank", body)
	if err != nil {
		return nil, err
	}

	var resp rerankResponse
	if err := json.Unmarshal(payload, &resp); err != nil {
		return nil, fmt.Errorf("voyageai rerank: decode response: %w", err)
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("voyageai rerank: no results for %d documents", len(req.Documents))
	}
	if len(resp.Data) > len(req.Documents) {
		return nil, fmt.Errorf("voyageai rerank: got %d results for %d documents", len(resp.Data), len(req.Documents))
	}

	// Every index is a position in the caller's slice, so it is validated
	// before it is used to read from that slice: a malformed or proxied
	// response must surface as an error, never as a panic or as a document
	// paired with the wrong score.
	results := make([]core.RerankResult, 0, len(resp.Data))
	seen := make(map[int]struct{}, len(resp.Data))
	for _, item := range resp.Data {
		if item.Index < 0 || item.Index >= len(req.Documents) {
			return nil, fmt.Errorf("voyageai rerank: result index %d out of range for %d documents", item.Index, len(req.Documents))
		}
		if _, dup := seen[item.Index]; dup {
			return nil, fmt.Errorf("voyageai rerank: duplicate result index %d", item.Index)
		}
		seen[item.Index] = struct{}{}

		results = append(results, core.RerankResult{
			Index:    item.Index,
			Score:    item.RelevanceScore,
			Document: req.Documents[item.Index],
		})
	}

	// Voyage returns results already sorted, but the contract is ours to keep,
	// so sort rather than trust: descending score, ties broken by ascending
	// request position to make the order total and reproducible.
	slices.SortStableFunc(results, func(a, b core.RerankResult) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		return cmp.Compare(a.Index, b.Index)
	})

	// Enforce TopN locally too. top_k is the server's job, but a proxy that
	// drops or ignores the field must not widen the result set past what the
	// caller asked for.
	if req.TopN > 0 && len(results) > req.TopN {
		results = results[:req.TopN]
	}

	model := resp.Model
	if model == "" {
		model = r.model
	}

	return &core.RerankResponse{
		Results: results,
		Model:   model,
		Usage:   core.RerankUsage{TotalTokens: resp.Usage.TotalTokens},
	}, nil
}

// Name implements core.Reranker.
func (r *Reranker) Name() string {
	return r.model
}

// Compile-time check that Reranker implements core.Reranker.
var _ core.Reranker = (*Reranker)(nil)
