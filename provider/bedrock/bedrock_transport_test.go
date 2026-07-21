package bedrock_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/provider/bedrock"
)

// converseBody captures the JSON body of a Converse call and replies with a
// minimal valid response, so assertions run against what the AWS SDK actually
// serializes rather than against the Go structs feeding it.
func converseBody(t *testing.T, req *core.ChatRequest) map[string]any {
	t.Helper()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode request body: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("x-amzn-RequestId", "req-transport-1")
		_, _ = w.Write([]byte(`{
			"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
			"stopReason": "end_turn",
			"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
			"metrics": {"latencyMs": 4}
		}`))
	}))
	t.Cleanup(srv.Close)

	client := bedrockruntime.New(bedrockruntime.Options{
		Region:       "us-east-1",
		BaseEndpoint: aws.String(srv.URL),
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
	})

	model, err := bedrock.New("anthropic.test", bedrock.WithClient(client))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := model.Request(context.Background(), req); err != nil {
		t.Fatalf("Request: %v", err)
	}
	if body == nil {
		t.Fatal("no request body captured")
	}
	return body
}

// Bedrock takes a single system field, so several system messages must arrive
// as one joined block rather than only the last one.
func TestTransportJoinsSystemMessages(t *testing.T) {
	body := converseBody(t, &core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "be terse"),
			core.NewTextMessage(core.RoleSystem, "answer in Spanish"),
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	})

	system, ok := body["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("expected one system block on the wire, got %#v", body["system"])
	}
	block, _ := system[0].(map[string]any)
	if block["text"] != "be terse\n\nanswer in Spanish" {
		t.Fatalf("expected joined system prompt, got %#v", block["text"])
	}
}

// The pinned SDK serializes a cachePoint content block, so the documented
// Bedrock cache-point support has to appear on the wire.
func TestTransportSendsCachePoint(t *testing.T) {
	body := converseBody(t, &core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			agentic.NewMultiPartMessage(
				agentic.TextPart("cache this"),
				agentic.CachePointPart("1h"),
			),
		},
	})

	messages, ok := body["messages"].([]any)
	if !ok || len(messages) != 1 {
		t.Fatalf("expected one message, got %#v", body["messages"])
	}
	content, _ := messages[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected text plus cachePoint blocks, got %#v", content)
	}

	cacheBlock, _ := content[1].(map[string]any)
	cachePoint, ok := cacheBlock["cachePoint"].(map[string]any)
	if !ok {
		t.Fatalf("expected a cachePoint block, got %#v", cacheBlock)
	}
	if cachePoint["type"] != "default" {
		t.Fatalf("expected default cache point type, got %#v", cachePoint["type"])
	}
	if cachePoint["ttl"] != "1h" {
		t.Fatalf("expected 1h ttl on the wire, got %#v", cachePoint["ttl"])
	}
}

// StopSequences is a universal request field and must reach Bedrock's
// inferenceConfig.
func TestTransportSendsStopSequences(t *testing.T) {
	body := converseBody(t, &core.ChatRequest{
		Model:         "anthropic.test",
		Messages:      []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		StopSequences: []string{"END"},
	})

	inference, ok := body["inferenceConfig"].(map[string]any)
	if !ok {
		t.Fatalf("expected inferenceConfig, got %#v", body["inferenceConfig"])
	}
	stops, _ := inference["stopSequences"].([]any)
	if len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("expected stop sequences on the wire, got %#v", inference["stopSequences"])
	}
}

// A forced tool choice is rejected by Bedrock when thinking is on, so it must
// leave as auto.
func TestTransportDowngradesForcedToolChoiceUnderThinking(t *testing.T) {
	required := core.ToolChoiceRequired
	body := converseBody(t, &core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
		Tools: []core.Tool{{
			Type:     core.ToolTypeFunction,
			Function: core.Function{Name: "lookup", Parameters: map[string]any{"type": "object"}},
		}},
		ToolChoice: &required,
		Thinking:   &core.ThinkingConfig{Enabled: true, BudgetTokens: 1024},
	})

	toolConfig, ok := body["toolConfig"].(map[string]any)
	if !ok {
		t.Fatalf("expected toolConfig, got %#v", body["toolConfig"])
	}
	choice, ok := toolConfig["toolChoice"].(map[string]any)
	if !ok {
		t.Fatalf("expected toolChoice, got %#v", toolConfig["toolChoice"])
	}
	if _, forced := choice["any"]; forced {
		t.Fatalf("expected forced choice to be downgraded, got %#v", choice)
	}
	if _, auto := choice["auto"]; !auto {
		t.Fatalf("expected toolChoice.auto, got %#v", choice)
	}
}

// The AWS request id is the only identifier Converse returns, so it stands in
// as the response id.
func TestTransportResponseCarriesRequestID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("x-amzn-RequestId", "req-transport-1")
		_, _ = w.Write([]byte(`{
			"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
			"stopReason": "model_context_window_exceeded",
			"usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
			"metrics": {"latencyMs": 4}
		}`))
	}))
	defer srv.Close()

	client := bedrockruntime.New(bedrockruntime.Options{
		Region:       "us-east-1",
		BaseEndpoint: aws.String(srv.URL),
		Credentials:  credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
	})
	model, err := bedrock.New("anthropic.test", bedrock.WithClient(client))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := model.Request(context.Background(), &core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if resp.ID != "req-transport-1" {
		t.Fatalf("expected response id from the AWS request id, got %q", resp.ID)
	}
	if resp.FinishReason != core.FinishReasonLength {
		t.Fatalf("expected context-window overflow to map to length, got %q", resp.FinishReason)
	}
	if resp.RawFinishReason != "model_context_window_exceeded" {
		t.Fatalf("expected the raw stop reason to pass through, got %q", resp.RawFinishReason)
	}
}

// The real SDK-backed stream client must surface a transport failure instead of
// returning a nil stream.
func TestTransportConverseStreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message": "ValidationException"}`))
	}))
	defer srv.Close()

	client := bedrockruntime.New(bedrockruntime.Options{
		Region:           "us-east-1",
		BaseEndpoint:     aws.String(srv.URL),
		Credentials:      credentials.NewStaticCredentialsProvider("AKID", "SECRET", ""),
		RetryMaxAttempts: 1,
	})
	model, err := bedrock.New("anthropic.test", bedrock.WithClient(client))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if _, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	}); err == nil {
		t.Fatal("expected a stream creation error")
	}
}
