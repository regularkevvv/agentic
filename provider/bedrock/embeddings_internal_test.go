package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	"github.com/regularkevvv/agentic/internal/retrieval"
)

// mockInvokeClient records every InvokeModel body it is handed and replies with
// canned responses, so tests can assert on the exact JSON that reaches Bedrock.
type mockInvokeClient struct {
	mu      sync.Mutex
	bodies  []json.RawMessage
	modelID string

	// responses are returned in call order. The final entry is reused once
	// exhausted, which keeps fan-out tests from having to size the slice.
	responses []string

	// byInput answers on the request body instead of on call order. Fan-out
	// calls complete in nondeterministic order, so a test asserting that
	// vectors come back in input order has to pin each response to its input.
	byInput map[string]string

	err error
}

func (m *mockInvokeClient) InvokeModel(
	ctx context.Context,
	params *bedrockruntime.InvokeModelInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.InvokeModelOutput, error) {
	m.mu.Lock()
	index := len(m.bodies)
	m.bodies = append(m.bodies, json.RawMessage(params.Body))
	m.modelID = *params.ModelId
	m.mu.Unlock()

	if m.err != nil {
		return nil, m.err
	}

	if m.byInput != nil {
		var req struct {
			InputText string `json:"inputText"`
		}
		if err := json.Unmarshal(params.Body, &req); err != nil {
			return nil, err
		}
		response, ok := m.byInput[req.InputText]
		if !ok {
			return nil, fmt.Errorf("mock has no response for %q", req.InputText)
		}
		return &bedrockruntime.InvokeModelOutput{Body: []byte(response)}, nil
	}

	if index >= len(m.responses) {
		index = len(m.responses) - 1
	}
	return &bedrockruntime.InvokeModelOutput{Body: []byte(m.responses[index])}, nil
}

// body returns the decoded request body of the nth InvokeModel call.
func (m *mockInvokeClient) body(t *testing.T, n int) map[string]any {
	t.Helper()

	m.mu.Lock()
	defer m.mu.Unlock()

	if n >= len(m.bodies) {
		t.Fatalf("wanted call %d, only %d calls were made", n, len(m.bodies))
	}
	var decoded map[string]any
	if err := json.Unmarshal(m.bodies[n], &decoded); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	return decoded
}

func (m *mockInvokeClient) calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.bodies)
}

// titanEmbedder builds a Titan Embedder wired to the given mock.
func titanEmbedder(modelID string, client invokeClient) *Embedder {
	family, version, _ := detectEmbeddingFamily(modelID)
	return &Embedder{client: client, modelID: modelID, family: family, version: version}
}

func TestStripGeoPrefix(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"us.amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v2:0"},
		{"eu.cohere.embed-v4:0", "cohere.embed-v4:0"},
		{"apac.cohere.embed-english-v3", "cohere.embed-english-v3"},
		{"jp.cohere.embed-v4:0", "cohere.embed-v4:0"},
		{"au.cohere.embed-v4:0", "cohere.embed-v4:0"},
		{"ca.cohere.embed-v4:0", "cohere.embed-v4:0"},
		{"global.cohere.embed-v4:0", "cohere.embed-v4:0"},
		{"us-gov.cohere.embed-v4:0", "cohere.embed-v4:0"},
		// No prefix, and a vendor segment that must not be mistaken for one.
		{"amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v2:0"},
		{"cohere.embed-english-v3", "cohere.embed-english-v3"},
	}

	for _, tt := range tests {
		if got := stripGeoPrefix(tt.in); got != tt.want {
			t.Errorf("stripGeoPrefix(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestModelVersion(t *testing.T) {
	tests := []struct {
		in   string
		want int
	}{
		{"amazon.titan-embed-text-v1", 1},
		{"amazon.titan-embed-text-v2:0", 2},
		{"cohere.embed-english-v3", 3},
		{"cohere.embed-v4:0", 4},
		// No version segment at all falls back to zero.
		{"amazon.titan-embed-text", 0},
		// A version too large for an int also falls back to zero.
		{"amazon.titan-embed-text-v99999999999999999999999", 0},
	}

	for _, tt := range tests {
		if got := modelVersion(tt.in); got != tt.want {
			t.Errorf("modelVersion(%q) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestDetectEmbeddingFamily(t *testing.T) {
	tests := []struct {
		modelID     string
		wantFamily  embeddingFamily
		wantVersion int
	}{
		{"amazon.titan-embed-text-v2:0", familyTitan, 2},
		{"amazon.titan-embed-text-v1", familyTitan, 1},
		{"cohere.embed-english-v3", familyCohere, 3},
		{"cohere.embed-v4:0", familyCohere, 4},
		// A geo prefix must not divert detection to the fallback branch.
		{"us.amazon.titan-embed-text-v2:0", familyTitan, 2},
		{"eu.cohere.embed-v4:0", familyCohere, 4},
	}

	for _, tt := range tests {
		family, version, err := detectEmbeddingFamily(tt.modelID)
		if err != nil {
			t.Errorf("detectEmbeddingFamily(%q): unexpected error %v", tt.modelID, err)
			continue
		}
		if family != tt.wantFamily || version != tt.wantVersion {
			t.Errorf("detectEmbeddingFamily(%q) = (%d, %d), want (%d, %d)",
				tt.modelID, family, version, tt.wantFamily, tt.wantVersion)
		}
	}
}

func TestDetectEmbeddingFamilyRejectsNova(t *testing.T) {
	_, _, err := detectEmbeddingFamily("amazon.nova-2-multimodal-embeddings-v1:0")
	if err == nil {
		t.Fatal("expected Nova to be rejected")
	}
	if !strings.Contains(err.Error(), "out of scope") {
		t.Errorf("expected an out-of-scope message, got %v", err)
	}
}

func TestDetectEmbeddingFamilyRejectsUnknown(t *testing.T) {
	_, _, err := detectEmbeddingFamily("meta.llama3-1-70b-instruct-v1:0")
	if err == nil {
		t.Fatal("expected an unknown model to be rejected")
	}
	if !strings.Contains(err.Error(), "unsupported model") {
		t.Errorf("expected an unsupported-model message, got %v", err)
	}
}

func TestCohereInputType(t *testing.T) {
	tests := []struct {
		in   retrieval.EmbeddingInputType
		want string
	}{
		{retrieval.EmbeddingInputQuery, "search_query"},
		{retrieval.EmbeddingInputDocument, "search_document"},
		// Cohere requires the field, so the no-preference case still has to
		// send something.
		{retrieval.EmbeddingInputNone, "search_document"},
	}

	for _, tt := range tests {
		if got := cohereInputType(tt.in); got != tt.want {
			t.Errorf("cohereInputType(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestCohereTruncate(t *testing.T) {
	yes, no := true, false

	if got := cohereTruncate(nil); got != "" {
		t.Errorf("cohereTruncate(nil) = %q, want the field omitted", got)
	}
	if got := cohereTruncate(&yes); got != "END" {
		t.Errorf("cohereTruncate(true) = %q, want END", got)
	}
	if got := cohereTruncate(&no); got != "NONE" {
		t.Errorf("cohereTruncate(false) = %q, want NONE", got)
	}
}

func TestInputTokenCountFrom(t *testing.T) {
	withHeader := func(value string) *smithyhttp.Response {
		header := http.Header{}
		if value != "" {
			header.Set(inputTokenCountHeader, value)
		}
		return &smithyhttp.Response{Response: &http.Response{Header: header}}
	}

	tests := []struct {
		name string
		raw  any
		want int
	}{
		{"header present", withHeader("42"), 42},
		{"header absent", withHeader(""), 0},
		{"header not a number", withHeader("many"), 0},
		{"header negative", withHeader("-5"), 0},
		{"no raw response at all", nil, 0},
		{"raw response of another type", "not a response", 0},
		{"typed nil response", (*smithyhttp.Response)(nil), 0},
		{"response without an http response", &smithyhttp.Response{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := inputTokenCountFrom(tt.raw); got != tt.want {
				t.Errorf("inputTokenCountFrom(%v) = %d, want %d", tt.raw, got, tt.want)
			}
		})
	}
}

// A stub client leaves the result metadata empty, which is the same shape a
// response stripped of its headers by a proxy would arrive in.
func TestInputTokenCountWithoutMetadata(t *testing.T) {
	if got := inputTokenCount(&bedrockruntime.InvokeModelOutput{}); got != 0 {
		t.Errorf("expected zero tokens without metadata, got %d", got)
	}
}

func TestEmbedTitanSendsOneCallPerInput(t *testing.T) {
	client := &mockInvokeClient{byInput: map[string]string{
		"first":  `{"embedding": [0.1, 0.2], "inputTextTokenCount": 3}`,
		"second": `{"embedding": [0.3, 0.4], "inputTextTokenCount": 4}`,
	}}
	embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input: []string{"first", "second"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if client.calls() != 2 {
		t.Fatalf("expected one call per input, got %d", client.calls())
	}
	if len(resp.Vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(resp.Vectors))
	}
	// Vectors must land in input order regardless of completion order.
	if resp.Vectors[0][0] != 0.1 || resp.Vectors[1][0] != 0.3 {
		t.Errorf("vectors out of input order: %v", resp.Vectors)
	}
	// The header is absent from the stub, so the body count stands in and is
	// summed across the fan-out.
	if resp.Usage.TotalTokens != 7 || resp.Usage.PromptTokens != 7 {
		t.Errorf("expected 7 tokens summed across calls, got %+v", resp.Usage)
	}
	if resp.Model != "amazon.titan-embed-text-v2:0" {
		t.Errorf("unexpected model %q", resp.Model)
	}
}

func TestEmbedTitanV2SendsDimensionsAndNormalize(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embedding": [0.1], "inputTextTokenCount": 1}`}}
	embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"hello"},
		Dimensions: 512,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	body := client.body(t, 0)
	if body["inputText"] != "hello" {
		t.Errorf("unexpected inputText %v", body["inputText"])
	}
	if body["dimensions"] != float64(512) {
		t.Errorf("expected dimensions 512, got %v", body["dimensions"])
	}
	if body["normalize"] != true {
		t.Errorf("expected normalize to default to true, got %v", body["normalize"])
	}
	// Titan takes a single text, never an array.
	if _, ok := body["texts"]; ok {
		t.Error("Titan body must not carry a texts array")
	}
}

func TestEmbedTitanHonorsNormalizeOption(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embedding": [0.1], "inputTextTokenCount": 1}`}}
	embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)
	normalize := false
	embedder.normalize = &normalize

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input: []string{"hello"},
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if body := client.body(t, 0); body["normalize"] != false {
		t.Errorf("expected normalize false, got %v", body["normalize"])
	}
}

// Titan v1 predates both parameters, so neither may appear on the wire.
func TestEmbedTitanV1OmitsDimensionsAndNormalize(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embedding": [0.1], "inputTextTokenCount": 1}`}}
	embedder := titanEmbedder("amazon.titan-embed-text-v1", client)

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input: []string{"hello"},
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	body := client.body(t, 0)
	if _, ok := body["dimensions"]; ok {
		t.Error("Titan v1 must not receive a dimensions parameter")
	}
	if _, ok := body["normalize"]; ok {
		t.Error("Titan v1 must not receive a normalize parameter")
	}
}

func TestEmbedTitanV1RejectsDimensions(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embedding": [0.1]}`}}
	embedder := titanEmbedder("amazon.titan-embed-text-v1", client)

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"hello"},
		Dimensions: 512,
	})
	if err == nil {
		t.Fatal("expected dimensions on Titan v1 to be rejected")
	}
	if client.calls() != 0 {
		t.Error("expected no request to be sent")
	}
}

func TestEmbedTitanRejectsTruncation(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embedding": [0.1]}`}}
	embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)
	truncate := true

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:    []string{"hello"},
		Truncate: &truncate,
	})
	if err == nil {
		t.Fatal("expected truncation to be rejected")
	}
	if client.calls() != 0 {
		t.Error("expected no request to be sent")
	}
}

// Truncate=false already matches Titan's behavior, so it must not error.
func TestEmbedTitanAcceptsTruncateFalse(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embedding": [0.1], "inputTextTokenCount": 1}`}}
	embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)
	truncate := false

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:    []string{"hello"},
		Truncate: &truncate,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
}

func TestEmbedTitanErrors(t *testing.T) {
	tests := []struct {
		name     string
		response string
		wantErr  string
	}{
		{"malformed json", `{"embedding": `, "decode Titan response"},
		{"empty embedding", `{"embedding": []}`, "carried no embedding"},
		{"missing embedding", `{"inputTextTokenCount": 3}`, "carried no embedding"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &mockInvokeClient{responses: []string{tt.response}}
			embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)

			_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected an error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestEmbedTitanPropagatesInvokeError(t *testing.T) {
	client := &mockInvokeClient{err: errors.New("throttled")}
	embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}})
	if err == nil || !strings.Contains(err.Error(), "throttled") {
		t.Fatalf("expected the underlying error to surface, got %v", err)
	}
}

func TestEmbedCohereSendsOneBatchedCall(t *testing.T) {
	client := &mockInvokeClient{responses: []string{
		`{"embeddings": [[0.1, 0.2], [0.3, 0.4]], "id": "abc"}`,
	}}
	embedder := titanEmbedder("cohere.embed-english-v3", client)

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:     []string{"first", "second"},
		InputType: retrieval.EmbeddingInputQuery,
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if client.calls() != 1 {
		t.Fatalf("expected a single batched call, got %d", client.calls())
	}
	body := client.body(t, 0)
	texts, ok := body["texts"].([]any)
	if !ok || len(texts) != 2 || texts[0] != "first" {
		t.Fatalf("unexpected texts %v", body["texts"])
	}
	if body["input_type"] != "search_query" {
		t.Errorf("expected search_query, got %v", body["input_type"])
	}
	// Truncate was nil, so the field is omitted and Cohere's default applies.
	if _, ok := body["truncate"]; ok {
		t.Error("truncate must be omitted when unset")
	}
	if len(resp.Vectors) != 2 || resp.Vectors[1][1] != 0.4 {
		t.Errorf("unexpected vectors %v", resp.Vectors)
	}
}

func TestEmbedCohereInputTypeMapping(t *testing.T) {
	tests := []struct {
		inputType retrieval.EmbeddingInputType
		want      string
	}{
		{retrieval.EmbeddingInputQuery, "search_query"},
		{retrieval.EmbeddingInputDocument, "search_document"},
		{retrieval.EmbeddingInputNone, "search_document"},
	}

	for _, tt := range tests {
		client := &mockInvokeClient{responses: []string{`{"embeddings": [[0.1]]}`}}
		embedder := titanEmbedder("cohere.embed-english-v3", client)

		if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
			Input:     []string{"hello"},
			InputType: tt.inputType,
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if got := client.body(t, 0)["input_type"]; got != tt.want {
			t.Errorf("input type %q mapped to %v, want %q", tt.inputType, got, tt.want)
		}
	}
}

func TestEmbedCohereTruncateMapping(t *testing.T) {
	yes, no := true, false

	for _, tt := range []struct {
		truncate *bool
		want     string
	}{{&yes, "END"}, {&no, "NONE"}} {
		client := &mockInvokeClient{responses: []string{`{"embeddings": [[0.1]]}`}}
		embedder := titanEmbedder("cohere.embed-english-v3", client)

		if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
			Input:    []string{"hello"},
			Truncate: tt.truncate,
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if got := client.body(t, 0)["truncate"]; got != tt.want {
			t.Errorf("truncate %v mapped to %v, want %q", *tt.truncate, got, tt.want)
		}
	}
}

func TestEmbedCohereV4SendsOutputDimension(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embeddings": [[0.1]]}`}}
	embedder := titanEmbedder("cohere.embed-v4:0", client)

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"hello"},
		Dimensions: 512,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}

	body := client.body(t, 0)
	if body["output_dimension"] != float64(512) {
		t.Errorf("expected output_dimension 512, got %v", body["output_dimension"])
	}
	// Cohere names the field output_dimension, not dimensions.
	if _, ok := body["dimensions"]; ok {
		t.Error("Cohere body must not carry a dimensions field")
	}
}

func TestEmbedCohereV3RejectsDimensions(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embeddings": [[0.1]]}`}}
	embedder := titanEmbedder("cohere.embed-english-v3", client)

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
		Input:      []string{"hello"},
		Dimensions: 512,
	})
	if err == nil {
		t.Fatal("expected dimensions on Cohere v3 to be rejected")
	}
	if client.calls() != 0 {
		t.Error("expected no request to be sent")
	}
}

// Cohere also returns embeddings keyed by type; only float vectors are asked
// for, so only that key is read.
func TestEmbedCohereEmbeddingsByType(t *testing.T) {
	client := &mockInvokeClient{responses: []string{
		`{"embeddings": {"float": [[0.1, 0.2]]}}`,
	}}
	embedder := titanEmbedder("cohere.embed-v4:0", client)

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Vectors) != 1 || resp.Vectors[0][1] != 0.2 {
		t.Errorf("unexpected vectors %v", resp.Vectors)
	}
}

func TestEmbedCohereErrors(t *testing.T) {
	tests := []struct {
		name     string
		response string
		wantErr  string
	}{
		{"malformed json", `{"embeddings": `, "decode Cohere response"},
		{"missing embeddings", `{"id": "abc"}`, "carried no embeddings field"},
		{"null embeddings", `{"embeddings": null}`, "carried no embeddings field"},
		{"embeddings of another type only", `{"embeddings": {"int8": [[1]]}}`, "no float embeddings"},
		{"undecodable embeddings", `{"embeddings": "nonsense"}`, "decode Cohere embeddings"},
		// A short batch would silently misalign the caller's join.
		{"too few vectors", `{"embeddings": [[0.1]]}`, "got 1 vectors for 2 inputs"},
		{"too many vectors", `{"embeddings": [[0.1], [0.2], [0.3]]}`, "got 3 vectors for 2 inputs"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &mockInvokeClient{responses: []string{tt.response}}
			embedder := titanEmbedder("cohere.embed-english-v3", client)

			_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{
				Input: []string{"first", "second"},
			})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected an error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestEmbedCoherePropagatesInvokeError(t *testing.T) {
	client := &mockInvokeClient{err: errors.New("throttled")}
	embedder := titanEmbedder("cohere.embed-english-v3", client)

	_, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}})
	if err == nil || !strings.Contains(err.Error(), "throttled") {
		t.Fatalf("expected the underlying error to surface, got %v", err)
	}
}

// Cohere caps a call at 96 texts, so a larger request is split and rejoined.
func TestEmbedCohereSplitsOversizeBatch(t *testing.T) {
	first := make([]string, 0, cohereBatchLimit)
	for i := range cohereBatchLimit {
		first = append(first, fmt.Sprintf("[%d]", i))
	}
	client := &mockInvokeClient{responses: []string{
		`{"embeddings": [` + strings.Join(first, ",") + `]}`,
		`{"embeddings": [[999]]}`,
	}}
	embedder := titanEmbedder("cohere.embed-english-v3", client)

	inputs := make([]string, cohereBatchLimit+1)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("text-%d", i)
	}

	resp, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: inputs})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if client.calls() != 2 {
		t.Fatalf("expected the batch to be split into 2 calls, got %d", client.calls())
	}
	if len(client.body(t, 0)["texts"].([]any)) != cohereBatchLimit {
		t.Error("first chunk should be filled to the limit")
	}
	if len(client.body(t, 1)["texts"].([]any)) != 1 {
		t.Error("second chunk should hold the remainder")
	}
	if len(resp.Vectors) != cohereBatchLimit+1 {
		t.Fatalf("expected %d vectors, got %d", cohereBatchLimit+1, len(resp.Vectors))
	}
	// The remainder must be appended after the first chunk, not interleaved.
	if resp.Vectors[cohereBatchLimit][0] != 999 {
		t.Errorf("chunks rejoined out of order: %v", resp.Vectors[cohereBatchLimit])
	}
}

func TestEmbedValidatesRequest(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embedding": [0.1]}`}}
	embedder := titanEmbedder("amazon.titan-embed-text-v2:0", client)

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: nil}); err == nil {
		t.Fatal("expected an empty input to be rejected")
	}
	if client.calls() != 0 {
		t.Error("expected no request to be sent")
	}
}

// The model ID travels to Bedrock exactly as configured, geo prefix included,
// since the prefix is what routes the call across regions.
func TestEmbedSendsPrefixedModelIDUnchanged(t *testing.T) {
	client := &mockInvokeClient{responses: []string{`{"embeddings": [[0.1]]}`}}
	embedder := titanEmbedder("us.cohere.embed-v4:0", client)

	if _, err := embedder.Embed(context.Background(), &retrieval.EmbeddingRequest{Input: []string{"hello"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if client.modelID != "us.cohere.embed-v4:0" {
		t.Errorf("expected the prefixed model ID on the wire, got %q", client.modelID)
	}
}
