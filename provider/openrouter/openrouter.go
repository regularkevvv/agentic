// Package openrouter provides an OpenRouter Model implementation for Agentic.
// OpenRouter is an API gateway that routes to many LLM providers (OpenAI, Anthropic,
// Google, Meta, etc.) through a unified OpenAI-compatible API.
//
// The wire format is OpenAI Chat Completions, but the gateway differs from
// OpenAI itself in ways that matter and are handled here:
//
//   - It accepts only the legacy max_tokens field, not max_completion_tokens.
//   - It reports upstream failures in band, with an HTTP 200 status and an
//     "error" object in the body.
//   - It surfaces reasoning text in a non-standard "reasoning" field.
//   - It accepts routing preferences in a top-level "provider" object.
//
// Because of those differences this package speaks to the API directly rather
// than delegating to the OpenAI provider.
package openrouter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// DefaultBaseURL is the OpenRouter API endpoint.
const DefaultBaseURL = "https://openrouter.ai/api/v1"

// Model implements core.Model and core.StreamModel against the OpenRouter API.
type Model struct {
	client *openai.Client
	model  string
}

// Option configures the OpenRouter Model.
type Option func(*config)

type config struct {
	apiKey      string
	baseURL     string
	httpReferer string
	appTitle    string
	extraOpts   []option.RequestOption
}

// WithAPIKey sets the API key. If not set, the OPENROUTER_API_KEY env var is used.
func WithAPIKey(apiKey string) Option {
	return func(c *config) { c.apiKey = apiKey }
}

// WithBaseURL overrides the default OpenRouter base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *config) { c.baseURL = baseURL }
}

// WithHTTPReferer sets the HTTP-Referer header for OpenRouter rankings.
// This helps OpenRouter attribute traffic to your site. If not set, the
// OPENROUTER_APP_URL env var is used.
func WithHTTPReferer(referer string) Option {
	return func(c *config) { c.httpReferer = referer }
}

// WithAppTitle sets the X-Title header shown in OpenRouter dashboards.
// If not set, the OPENROUTER_APP_TITLE env var is used.
func WithAppTitle(title string) Option {
	return func(c *config) { c.appTitle = title }
}

// WithRequestOptions adds raw SDK request options (custom headers, a custom
// HTTP client, middleware). Options are applied after the ones this package
// derives from the other Option values, so they win on any conflict.
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(c *config) { c.extraOpts = append(c.extraOpts, opts...) }
}

// New creates a new OpenRouter Model.
//
// Example:
//
//	model, err := openrouter.New("anthropic/claude-sonnet-4", openrouter.WithAPIKey("sk-or-..."))
//	model, err := openrouter.New("openai/gpt-4o")  // uses OPENROUTER_API_KEY env var
func New(model string, opts ...Option) (*Model, error) {
	cfg := &config{}
	for _, opt := range opts {
		opt(cfg)
	}

	apiKey := cfg.apiKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("openrouter: API key not set (use WithAPIKey or set OPENROUTER_API_KEY)")
	}

	baseURL := cfg.baseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	referer := cfg.httpReferer
	if referer == "" {
		referer = os.Getenv("OPENROUTER_APP_URL")
	}
	title := cfg.appTitle
	if title == "" {
		title = os.Getenv("OPENROUTER_APP_TITLE")
	}

	reqOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	if referer != "" {
		reqOpts = append(reqOpts, option.WithHeader("HTTP-Referer", referer))
	}
	if title != "" {
		reqOpts = append(reqOpts, option.WithHeader("X-Title", title))
	}
	reqOpts = append(reqOpts, cfg.extraOpts...)

	client := openai.NewClient(reqOpts...)

	return &Model{client: &client, model: model}, nil
}

// MustNew is like New but panics on error.
func MustNew(model string, opts ...Option) *Model {
	m, err := New(model, opts...)
	if err != nil {
		panic(err)
	}
	return m
}

// Name implements core.Model.
func (m *Model) Name() string {
	return m.model
}

// Request implements core.Model.
//
// OpenRouter reports upstream failures in band: an HTTP 200 response whose body
// carries an "error" object and a null choices array. Request re-decodes the raw
// body to detect that and returns an *APIError. In that case the returned
// response is non-nil and carries FinishReasonError along with the gateway's
// raw stop reason, so a caller that inspects the response after logging the
// error still sees why the turn ended. Callers should check the error first.
func (m *Model) Request(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := m.buildParams(req)

	resp, err := m.client.Chat.Completions.New(ctx, params, requestOptions(req)...)
	if err != nil {
		return nil, err
	}

	if apiErr := decodeInBandError(resp.RawJSON()); apiErr != nil {
		return &core.ChatResponse{
			ID:              resp.ID,
			Model:           string(resp.Model),
			Message:         core.Message{Role: core.RoleAssistant, Content: []core.Part{}},
			Created:         time.Unix(resp.Created, 0),
			FinishReason:    core.FinishReasonError,
			RawFinishReason: rawFinishReason(resp),
		}, apiErr
	}

	if len(resp.Choices) == 0 {
		return nil, ErrNoChoices
	}

	choice := resp.Choices[0]

	return &core.ChatResponse{
		ID:              resp.ID,
		Model:           string(resp.Model),
		Message:         convertResponseMessage(choice.Message),
		Usage:           extractUsage(resp.Usage),
		Created:         time.Unix(resp.Created, 0),
		FinishReason:    convertFinishReason(choice.FinishReason),
		RawFinishReason: choice.FinishReason,
	}, nil
}

// RequestStream implements core.StreamModel.
//
// An in-band error object delivered mid-stream aborts the stream and is
// surfaced as a StreamEventError carrying an *APIError, mirroring the
// non-streaming path.
func (m *Model) RequestStream(ctx context.Context, req *core.ChatRequest) (*core.StreamResult, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	params := m.buildParams(req)
	// OpenRouter only reports usage on a streamed turn when it is asked to.
	// Mirrors pydantic-ai models/openai.py:1262.
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}

	stream := m.client.Chat.Completions.NewStreaming(ctx, params, requestOptions(req)...)

	ch := make(chan core.StreamEvent, 64)
	sr := core.NewStreamResult(ch)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		// Tool call IDs arrive only on the first chunk for a given index, so
		// later argument deltas have to be correlated back to it.
		seenToolCalls := make(map[int64]string)
		var usage core.Usage
		var rawFinish string

		for stream.Next() {
			chunk := stream.Current()

			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				usage = extractUsage(chunk.Usage)
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			if choice.FinishReason != "" {
				rawFinish = choice.FinishReason
			}

			// Reasoning text arrives in a non-standard delta field.
			if id, text := extractReasoning(choice.Delta.JSON.ExtraFields); text != "" {
				ch <- core.StreamEvent{
					Type:         core.StreamEventThinkingDelta,
					Delta:        text,
					ThinkingID:   id,
					ProviderName: ProviderName,
				}
			}

			if choice.Delta.Content != "" {
				ch <- core.StreamEvent{
					Type:  core.StreamEventTextDelta,
					Delta: choice.Delta.Content,
				}
			}

			for _, tc := range choice.Delta.ToolCalls {
				id, seen := seenToolCalls[tc.Index]
				if !seen {
					id = tc.ID
					seenToolCalls[tc.Index] = id
					ch <- core.StreamEvent{
						Type: core.StreamEventToolCallStart,
						ToolUse: &core.ToolUse{
							ID:   tc.ID,
							Name: tc.Function.Name,
						},
					}
				}

				if tc.Function.Arguments != "" {
					ch <- core.StreamEvent{
						Type:       core.StreamEventToolCallDelta,
						Delta:      tc.Function.Arguments,
						ToolCallID: id,
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- core.StreamEvent{Type: core.StreamEventError, Error: streamError(err)}
			return
		}

		ch <- core.StreamEvent{
			Type:         core.StreamEventDone,
			Usage:        &usage,
			FinishReason: convertFinishReason(rawFinish),
		}
	}()

	return sr, nil
}

// buildParams converts a core.ChatRequest into OpenRouter chat completion params.
func (m *Model) buildParams(req *core.ChatRequest) openai.ChatCompletionNewParams {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages))
	for _, msg := range req.Messages {
		messages = append(messages, convertMessage(msg))
	}

	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    shared.ChatModel(m.model),
	}
	if req.PromptCache != nil && req.PromptCache.Enabled() {
		params.PromptCacheKey = openai.String(promptCacheKey(req.PromptCache.Key))
	}

	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.MaxTokens != nil {
		// OpenRouter accepts only the legacy max_tokens field; sending
		// max_completion_tokens leaves the cap silently unapplied. See
		// pydantic-ai providers/openrouter.py:182
		// (openai_chat_supports_max_completion_tokens=False).
		params.MaxTokens = openai.Int(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.Float(*req.TopP)
	}
	if len(req.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: req.StopSequences,
		}
	}

	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
		if req.ToolChoice != nil {
			params.ToolChoice = convertToolChoice(*req.ToolChoice)
		}
	}

	if req.ResponseFormat != nil {
		params.ResponseFormat = convertResponseFormat(req.ResponseFormat)
	}

	return params
}

func promptCacheKey(value string) string {
	runes := []rune(value)
	if len(runes) > 64 {
		runes = runes[:64]
	}
	return string(runes)
}

// Compile-time checks that Model implements both core interfaces.
var (
	_ core.Model       = (*Model)(nil)
	_ core.StreamModel = (*Model)(nil)
)

// ErrNoChoices is returned when OpenRouter answers with a well-formed body that
// carries neither a choice nor an error object.
var ErrNoChoices = errors.New("openrouter: response contained no choices")

// APIError reports a failure OpenRouter delivered inside the response body
// rather than through the HTTP status, which the gateway does for upstream
// provider failures: the status is 200, choices is null, and the body carries
// an "error" object.
type APIError struct {
	// Code is the gateway's error code. It mirrors an HTTP status for
	// upstream failures (for example 429 or 502). Zero when absent.
	Code int
	// Message is the gateway's human-readable description of the failure.
	Message string
	// Metadata holds any provider-specific detail the gateway attached.
	Metadata map[string]interface{}
}

func (e *APIError) Error() string {
	if e.Code != 0 {
		return fmt.Sprintf("openrouter: upstream error %d: %s", e.Code, e.Message)
	}
	return "openrouter: upstream error: " + e.Message
}

// errorBody models OpenRouter's error object. Code is held undecoded because
// the gateway is not consistent about quoting it, and a code this library
// cannot parse must not cost the caller the message next to it. See pydantic-ai
// models/openrouter.py:342 (_OpenRouterError) and :570
// (_OpenRouterErrorResponse), which handles the same null-choices shape.
type errorBody struct {
	Code     json.RawMessage        `json:"code"`
	Message  string                 `json:"message"`
	Metadata map[string]interface{} `json:"metadata"`
}

// toAPIError converts a decoded error object, returning nil when it carries no
// information: a present-but-empty object must not fail a response on nothing.
func (b *errorBody) toAPIError() *APIError {
	if b == nil {
		return nil
	}
	code := parseErrorCode(b.Code)
	if b.Message == "" && code == 0 {
		return nil
	}
	return &APIError{
		Code:     code,
		Message:  b.Message,
		Metadata: b.Metadata,
	}
}

// parseErrorCode reads an error code that may arrive as a number or as a
// quoted number. Anything else yields 0, leaving the message to carry the
// detail rather than discarding the whole error.
func parseErrorCode(raw json.RawMessage) int {
	if len(raw) == 0 {
		return 0
	}
	var num json.Number
	if err := json.Unmarshal(raw, &num); err == nil {
		if code, err := num.Int64(); err == nil {
			return int(code)
		}
	}
	return 0
}

// decodeInBandError re-decodes a raw response body and returns an *APIError
// when the gateway reported a failure in band. A body that does not carry a
// usable error object yields nil, so a malformed or absent body is never
// mistaken for a failure.
func decodeInBandError(raw string) *APIError {
	if raw == "" {
		return nil
	}
	var env struct {
		Error *errorBody `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &env); err != nil {
		return nil
	}
	return env.Error.toAPIError()
}

// sdkStreamErrorPrefix is what the OpenAI SDK's SSE decoder puts in front of an
// error object it finds mid-stream. It aborts the stream itself and reports the
// object only as text (openai-go packages/ssestream/ssestream.go:171), so the
// typed error has to be recovered from the message.
const sdkStreamErrorPrefix = "received error while streaming: "

// streamError upgrades a stream failure into an *APIError when it carries
// OpenRouter's in-band error object, and otherwise wraps it as-is.
func streamError(err error) error {
	msg := err.Error()
	i := strings.Index(msg, sdkStreamErrorPrefix)
	if i < 0 {
		return fmt.Errorf("openrouter stream: %w", err)
	}

	var body errorBody
	if jsonErr := json.Unmarshal([]byte(msg[i+len(sdkStreamErrorPrefix):]), &body); jsonErr != nil {
		return fmt.Errorf("openrouter stream: %w", err)
	}
	if apiErr := body.toAPIError(); apiErr != nil {
		return apiErr
	}
	return fmt.Errorf("openrouter stream: %w", err)
}

// rawFinishReason returns the gateway's stop reason for an errored response,
// falling back to "error" when choices is null and none was reported.
func rawFinishReason(resp *openai.ChatCompletion) string {
	if len(resp.Choices) > 0 && resp.Choices[0].FinishReason != "" {
		return resp.Choices[0].FinishReason
	}
	return "error"
}
