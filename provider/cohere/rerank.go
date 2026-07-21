package cohere

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"

	"github.com/regularkevvv/agentic/internal/core"
)

// DefaultRerankModel is the model used by the examples in this package and the
// one Cohere currently recommends for new reranking work.
const DefaultRerankModel = "rerank-v3.5"

// Reranker implements core.Reranker using the Cohere /v2/rerank API.
type Reranker struct {
	model  string
	client *client
}

// RerankerOption configures the Cohere Reranker. It is deliberately distinct
// from Option so that a knob meaningful only to one endpoint cannot be passed
// to the constructor for the other.
type RerankerOption func(*rerankConfig)

type rerankConfig struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	maxRetries *int
}

// WithRerankerAPIKey sets the API key. If not set, the CO_API_KEY environment
// variable is used, then COHERE_API_KEY.
func WithRerankerAPIKey(apiKey string) RerankerOption {
	return func(c *rerankConfig) { c.apiKey = apiKey }
}

// WithRerankerBaseURL sets a custom base URL (default https://api.cohere.com).
// The API version is part of the request path, so the base URL must not include
// it.
func WithRerankerBaseURL(baseURL string) RerankerOption {
	return func(c *rerankConfig) { c.baseURL = baseURL }
}

// WithRerankerHTTPClient sets a custom HTTP client.
func WithRerankerHTTPClient(httpClient *http.Client) RerankerOption {
	return func(c *rerankConfig) { c.httpClient = httpClient }
}

// WithRerankerMaxRetries sets how many times a request is retried on 429 and
// 5xx responses (default 2, matching the SDK-backed providers).
func WithRerankerMaxRetries(retries int) RerankerOption {
	return func(c *rerankConfig) { c.maxRetries = &retries }
}

// NewReranker creates a new Cohere Reranker.
//
// Examples:
//
//	reranker, err := cohere.NewReranker("rerank-v3.5", cohere.WithRerankerAPIKey("..."))
//	reranker, err := cohere.NewReranker("rerank-english-v3.0") // key from CO_API_KEY
func NewReranker(model string, opts ...RerankerOption) (*Reranker, error) {
	cfg := &rerankConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	if model == "" {
		return nil, fmt.Errorf("cohere: model cannot be empty")
	}

	c, err := newClient(cfg.apiKey, cfg.baseURL, cfg.httpClient, cfg.maxRetries)
	if err != nil {
		return nil, err
	}

	return &Reranker{model: model, client: c}, nil
}

// MustNewReranker is like NewReranker but panics on error.
func MustNewReranker(model string, opts ...RerankerOption) *Reranker {
	r, err := NewReranker(model, opts...)
	if err != nil {
		panic(err)
	}
	return r
}

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

// rerankResponse declares only the fields this package maps. Cohere also
// returns an id, which core.RerankResponse has nowhere to put.
type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
	Meta meta `json:"meta"`
}

// Rerank implements core.Reranker.
//
// return_documents is deliberately not sent: the documents are already in the
// caller's hands, echoing them back doubles the size of the response for no
// gain, and RerankResult.Document is filled from the request slice regardless —
// so what a caller reads back is always the text they sent, never a value the
// API may have normalized or truncated.
func (r *Reranker) Rerank(ctx context.Context, req *core.RerankRequest) (*core.RerankResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	// TopN larger than the document count is clamped, as core.RerankRequest
	// documents; zero is omitted, which asks for a result per document.
	topN := req.TopN
	if topN > len(req.Documents) {
		topN = len(req.Documents)
	}

	body := rerankRequest{
		Model:     r.model,
		Query:     req.Query,
		Documents: req.Documents,
		TopN:      topN,
	}

	payload, err := r.client.post(ctx, "/v2/rerank", body)
	if err != nil {
		return nil, err
	}

	var resp rerankResponse
	if err := json.Unmarshal(payload, &resp); err != nil {
		return nil, fmt.Errorf("cohere rerank: decode response: %w", err)
	}

	results := make([]core.RerankResult, 0, len(resp.Results))
	for _, item := range resp.Results {
		// Bounds-check every index before using it to read the request slice.
		// The index is chosen by the server, so a malformed reply or a proxy
		// that reorders or pads results must produce an error here rather than
		// panicking inside the caller's request.
		if item.Index < 0 || item.Index >= len(req.Documents) {
			return nil, fmt.Errorf("cohere rerank: result index %d out of range for %d documents", item.Index, len(req.Documents))
		}
		results = append(results, core.RerankResult{
			Index:    item.Index,
			Score:    item.RelevanceScore,
			Document: req.Documents[item.Index],
		})
	}

	// Sort by descending score rather than trusting the response order. The API
	// documents that it returns results ranked, but core.RerankResponse promises
	// descending score to every caller, and that promise should not rest on a
	// remote service's ordering surviving intact through whatever sits between.
	// The sort is stable, so results the model scored identically keep the
	// relative order the API gave them.
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return &core.RerankResponse{
		Results: results,
		// Cohere does not echo the model back, so the configured name is
		// reported.
		Model: r.model,
		// Cohere meters reranking in search units — one query against a batch of
		// documents — not in tokens, so TotalTokens is left zero rather than
		// filled with a number the bill does not reflect.
		Usage: core.RerankUsage{SearchUnits: resp.Meta.BilledUnits.SearchUnits},
	}, nil
}

// Name implements core.Reranker.
func (r *Reranker) Name() string {
	return r.model
}

// Compile-time check that Reranker implements core.Reranker.
var _ core.Reranker = (*Reranker)(nil)
