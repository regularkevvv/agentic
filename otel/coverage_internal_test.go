package agenticotel

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

func TestContentProjectionCoversEveryAgenticPart(t *testing.T) {
	instrumentation := MustNew(
		WithMessageContent(),
		WithToolContent(),
		WithMaxContentBytes(1_000_000),
	)
	parts := []agentic.Part{
		{Type: agentic.ContentText, Text: "text"},
		{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{Text: "thought"}},
		{Type: agentic.ContentThinking, Thinking: &agentic.ThinkingBlock{ID: "redacted_thinking", Signature: "ciphertext"}},
		{Type: agentic.ContentToolUse, ToolUse: &agentic.ToolUse{ID: "call", Name: "tool", Input: map[string]any{"n": 1}}},
		{Type: agentic.ContentToolResult, ToolResult: &agentic.ToolResult{ToolUseID: "call", Content: "result"}},
		{Type: agentic.ContentImageURL, ImageURL: &agentic.ImageURL{URL: "https://image", MediaType: "image/png"}},
		{Type: agentic.ContentAudioURL, AudioURL: &agentic.AudioURL{URL: "https://audio", Format: "mp3"}},
		{Type: agentic.ContentVideoURL, VideoURL: &agentic.VideoURL{URL: "https://video", MediaType: "video/mp4"}},
		{Type: agentic.ContentDocumentURL, DocumentURL: &agentic.DocumentURL{URL: "https://doc", MediaType: "application/pdf"}},
		{Type: agentic.ContentUploadedFile, UploadedFile: &agentic.UploadedFile{FileID: "file-1"}},
		{Type: agentic.ContentImageData, ImageData: &agentic.ImageData{Data: "never-export", MediaType: "image/png"}},
		{Type: agentic.ContentCachePoint, CachePoint: &agentic.CachePoint{TTL: "5m"}},
		{Type: agentic.ContentImageURL},
		{Type: agentic.ContentAudioURL},
		{Type: agentic.ContentVideoURL},
		{Type: agentic.ContentDocumentURL},
		{Type: agentic.ContentUploadedFile},
		{Type: agentic.ContentToolUse},
		{Type: agentic.ContentToolResult},
		{Type: agentic.ContentThinking},
	}
	kept := 0
	for _, part := range parts {
		if _, ok := instrumentation.messagePart(part); ok {
			kept++
		}
	}
	if kept != 9 {
		t.Fatalf("kept parts = %d, want 9 safe structured parts", kept)
	}

	messages := []agentic.Message{
		{Role: agentic.RoleSystem, Content: []agentic.Part{{Type: agentic.ContentText, Text: "system"}}},
		{Role: agentic.RoleUser, Content: []agentic.Part{{Type: agentic.ContentText, Text: "user"}}},
		{Role: agentic.RoleAssistant, Content: []agentic.Part{{Type: agentic.ContentToolUse, ToolUse: &agentic.ToolUse{Name: "tool", Input: map[string]any{}}}}},
	}
	system, regular := instrumentation.messageContent(messages)
	if len(system) != 1 || len(regular) != 2 || normalizedFinishReason(agentic.FinishReasonToolCalls) != "tool_call" || normalizedFinishReason(agentic.FinishReasonStop) != "stop" {
		t.Fatalf("message projection = %#v / %#v", system, regular)
	}
	output := instrumentation.outputMessageContent([]agentic.Message{
		messages[2],
		agentic.NewToolResultMessage("call", "result", false),
		agentic.NewTextMessage(agentic.RoleAssistant, "final"),
	}, agentic.FinishReasonStop)
	if len(output) != 1 || output[0].(map[string]any)["finish_reason"] != "stop" {
		t.Fatalf("output projection = %#v", output)
	}
	toolOutput := instrumentation.outputMessageContent([]agentic.Message{messages[2]}, agentic.FinishReasonToolCalls)
	if len(toolOutput) != 1 || toolOutput[0].(map[string]any)["finish_reason"] != "tool_call" {
		t.Fatalf("tool-call output projection = %#v", toolOutput)
	}
	if output := instrumentation.outputMessageContent(messages, ""); output != nil {
		t.Fatalf("output without finish reason = %#v", output)
	}

	definitions := instrumentation.toolDefinitions([]agentic.Tool{{
		Type:     agentic.ToolTypeFunction,
		Function: agentic.Function{Name: "tool", Description: "description", Parameters: map[string]any{"type": "object"}},
	}})
	if len(definitions) != 1 {
		t.Fatalf("tool definitions = %#v", definitions)
	}
	if _, ok := instrumentation.contentAttribute(keyToolDefinitions, definitions); !ok {
		t.Fatal("valid structured content was rejected")
	}
	if _, ok := instrumentation.contentAttribute("bad", make(chan int)); ok {
		t.Fatal("unserializable content was accepted")
	}
	if _, ok := instrumentation.contentAttribute("nil", nil); ok {
		t.Fatal("null top-level content was accepted")
	}

	rejecting := &Instrumentation{config: config{filter: func(ContentKind, string) (string, bool) { return "", false }}}
	if _, ok := rejecting.messagePart(agentic.Part{Type: agentic.ContentText, Text: "x"}); ok {
		t.Fatal("rejected content was retained")
	}
	if _, ok := rejecting.filteredJSONObject(ContentToolArguments, map[string]any{"x": 1}); ok {
		t.Fatal("rejected JSON was retained")
	}
	rejecting.config.toolContent = true
	if _, ok := rejecting.messagePart(parts[3]); ok {
		t.Fatal("rejected tool arguments were retained")
	}
	if _, ok := rejecting.messagePart(parts[4]); ok {
		t.Fatal("rejected tool result was retained")
	}
	if _, ok := rejecting.uriPart("image", "https://private", "image/png"); ok {
		t.Fatal("rejected URI was retained")
	}
	if _, ok := rejecting.messagePart(agentic.Part{Type: agentic.ContentUploadedFile, UploadedFile: &agentic.UploadedFile{FileID: "private"}}); ok {
		t.Fatal("rejected file ID was retained")
	}
	if definitions := rejecting.toolDefinitions([]agentic.Tool{{Type: agentic.ToolTypeFunction, Function: agentic.Function{Name: "tool", Parameters: map[string]any{"type": "object"}}}}); len(definitions) != 1 {
		t.Fatalf("rejected tool parameters removed the definition: %#v", definitions)
	} else if _, exists := definitions[0].(map[string]any)["parameters"]; exists {
		t.Fatal("rejected tool parameters were retained")
	}
	if _, ok := instrumentation.uriPart("image", " data:image/png;base64,private", "image/png"); ok {
		t.Fatal("inline data URI was retained")
	}
	panicking := &Instrumentation{config: config{filter: func(ContentKind, string) (string, bool) { panic("filter") }}}
	if _, ok := panicking.filtered(ContentMessageText, "x"); ok {
		t.Fatal("panicking filter was not contained")
	}
	invalidJSON := &Instrumentation{config: config{filter: func(ContentKind, string) (string, bool) { return "not-json", true }}}
	if _, ok := invalidJSON.filteredJSONObject(ContentToolArguments, map[string]any{"x": 1}); ok {
		t.Fatal("invalid filtered JSON object was retained")
	}
	if _, ok := invalidJSON.filteredJSONObject(ContentToolArguments, make(chan int)); ok {
		t.Fatal("unserializable filter input was accepted")
	}
	for _, value := range []any{"plain text", []any{"array"}, 1, nil} {
		if _, ok := instrumentation.filteredJSONObject(ContentToolResult, value); ok {
			t.Fatalf("non-object tool value %#v was accepted", value)
		}
	}
	for _, value := range []any{`{"value":1}`, json.RawMessage(`{"value":2}`), []byte(`{"value":3}`)} {
		if _, ok := instrumentation.filteredJSONObject(ContentToolResult, value); !ok {
			t.Fatalf("serialized object %#v was rejected", value)
		}
	}
	withoutTools := &Instrumentation{config: config{messageContent: true}}
	if _, ok := withoutTools.messagePart(parts[1]); !ok {
		t.Fatal("reasoning disappeared with message capture enabled")
	}
	if _, ok := withoutTools.messagePart(parts[3]); ok {
		t.Fatal("tool arguments appeared without tool capture")
	}
	_, emptyMessages := instrumentation.messageContent([]agentic.Message{{Role: agentic.RoleUser, Content: []agentic.Part{{Type: agentic.ContentCachePoint}}}})
	if len(emptyMessages) != 0 {
		t.Fatal("message with no exportable parts was retained")
	}
}

func TestAttributeBuildersAndLowCardinalityErrors(t *testing.T) {
	max := 100
	temperature := 0.2
	topP := 0.8
	request := agentic.ChatRequest{
		MaxTokens: &max, Temperature: &temperature, TopP: &topP,
		StopSequences: []string{"stop"}, Stream: true,
		ResponseFormat: &agentic.ResponseFormat{Type: "json_object"},
	}
	if len(requestAttrs(request)) != 6 {
		t.Fatalf("request attrs = %#v", requestAttrs(request))
	}
	response := &agentic.ChatResponse{
		ID: "id", Model: "actual", FinishReason: agentic.FinishReasonStop,
		Usage: agentic.Usage{PromptTokens: 1, CompletionTokens: 2, CacheReadTokens: 3, CacheCreationTokens: 4, ReasoningTokens: 5},
	}
	if len(responseAttrs(response)) != 8 || responseAttrs(nil) != nil {
		t.Fatalf("response attrs = %#v", responseAttrs(response))
	}
	for _, test := range []struct {
		err  error
		want string
	}{
		{context.Canceled, "canceled"},
		{context.DeadlineExceeded, "deadline_exceeded"},
		{errors.New("message"), "errors.errorString"},
		{errors.Join(errors.New("first"), context.Canceled), "canceled"},
		{errors.Join(errors.New("first"), errors.New("second")), "errors.errorString"},
		{nil, ""},
	} {
		if got := errorType(test.err); got != test.want {
			t.Errorf("errorType(%v) = %q, want %q", test.err, got, test.want)
		}
	}
	values := []any{nil, "s", true, int(1), int64(2), float64(3), float32(4), []any{"x", 1.0}, map[string]any{"b": true, "a": 1.0}, struct{ X int }{1}}
	for _, value := range values {
		_ = valueFromAny(value)
	}
	for status, want := range map[agentic.ExecutionStatus]string{
		agentic.ExecutionCompleted: "completed", agentic.ExecutionSuspended: "suspended",
		agentic.ExecutionStopped: "stopped", agentic.ExecutionInterrupted: "interrupted",
		agentic.ExecutionFailed: "failed", agentic.ExecutionStatus(99): "unknown",
	} {
		if got := executionOutcome(status); got != want {
			t.Errorf("executionOutcome(%d) = %q", status, got)
		}
	}
	attrs := []attribute.KeyValue{}
	stringAttr("empty", "", &attrs)
	if len(attrs) != 0 {
		t.Fatal("empty optional string emitted")
	}
	if got := markSpanError(trace.SpanFromContext(context.Background()), nil, false); got != "" {
		t.Fatalf("nil error type = %q", got)
	}
}

type internalEmbedder struct {
	response *agentic.EmbeddingResponse
	err      error
	metadata agentic.ModelMetadata
}

func (e *internalEmbedder) Name() string                         { return "embed-model" }
func (e *internalEmbedder) ModelMetadata() agentic.ModelMetadata { return e.metadata }
func (e *internalEmbedder) Embed(context.Context, *agentic.EmbeddingRequest) (*agentic.EmbeddingResponse, error) {
	return e.response, e.err
}

func TestOptionsDirectLifecycleAndEmbeddingEdges(t *testing.T) {
	for name, option := range map[string]Option{
		"tracer": WithTracerProvider(nil),
		"meter":  WithMeterProvider(nil),
		"logger": WithLoggerProvider(nil),
		"filter": WithContentFilter(nil),
	} {
		if _, err := New(option); err == nil {
			t.Errorf("%s nil option accepted", name)
		}
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("MustNew(nil) did not panic")
			}
		}()
		_ = MustNew(nil)
	}()

	instrumentation := MustNew(WithMessageContent(), WithToolContent(), WithInferenceDetails())
	agentCtx, agentLifecycle := instrumentation.StartAgent(context.Background(), agentic.AgentOperation{
		Agent: agentic.AgentIdentity{Name: "direct"}, ModelName: "model", Mode: agentic.AgentInvocationResume,
		Input: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "input")},
	})
	modelCtx, modelLifecycle := instrumentation.StartModelRequest(agentCtx, agentic.ModelOperation{
		Agent: agentic.AgentIdentity{Name: "direct"},
		Model: agentic.ModelMetadata{Provider: "local", Operation: "chat", InProcess: true},
		Request: agentic.ChatRequest{
			Model:    "model",
			Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "input")},
			Tools:    []agentic.Tool{{Type: agentic.ToolTypeFunction, Function: agentic.Function{Name: "tool"}}},
		},
	})
	modelLifecycle.ObserveStreamEvent(agentic.StreamEvent{Type: agentic.StreamEventDone})
	modelLifecycle.End(agentic.ModelOperationResult{Error: errors.New("inference")})
	_, toolLifecycle := instrumentation.StartTool(modelCtx, agentic.ToolOperation{
		Agent:      agentic.AgentIdentity{Name: "direct"},
		Call:       agentic.ToolUse{ID: "id", Name: "tool", Input: map[string]any{"x": 1}},
		Definition: agentic.Tool{Type: agentic.ToolTypeFunction, Function: agentic.Function{Name: "tool", Description: "desc"}},
	})
	toolLifecycle.End(agentic.ToolOperationResult{Result: agentic.ToolExecutionResult{IsError: true, Content: "failed"}})
	agentLifecycle.End(agentic.AgentOperationResult{Status: agentic.ExecutionSuspended})
	_, failedAgent := instrumentation.StartAgent(context.Background(), agentic.AgentOperation{})
	failedAgent.End(agentic.AgentOperationResult{Status: agentic.ExecutionFailed})
	_, failedTool := instrumentation.StartTool(context.Background(), agentic.ToolOperation{Call: agentic.ToolUse{Name: "failed"}})
	failedTool.End(agentic.ToolOperationResult{Error: errors.New("tool failed")})
	instrumentation.emitException(context.Background(), nil)
	if err := instrumentation.RecordEvaluation(context.Background(), EvaluationResult{Name: "failure", Error: errors.New("bad evaluation")}); err != nil {
		t.Fatal(err)
	}

	embedder := &internalEmbedder{
		metadata: agentic.ModelMetadata{Provider: "local", InProcess: true},
		response: &agentic.EmbeddingResponse{Model: "actual", Usage: agentic.EmbeddingUsage{PromptTokens: 2}},
	}
	wrapped, err := instrumentation.WrapEmbedder(embedder)
	if err != nil || wrapped.Name() != "embed-model" {
		t.Fatalf("WrapEmbedder = %#v, %v", wrapped, err)
	}
	if _, err := wrapped.Embed(context.Background(), &agentic.EmbeddingRequest{Input: []string{"x"}}); err != nil {
		t.Fatal(err)
	}
	embedder.response, embedder.err = nil, context.Canceled
	if _, err := wrapped.Embed(context.Background(), &agentic.EmbeddingRequest{Input: []string{"x"}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("wrapped embedding error = %v", err)
	}
	if _, err := instrumentation.WrapEmbedder(embedder, nil); err == nil {
		t.Fatal("nil embedder option accepted")
	}
	if _, err := instrumentation.WrapEmbedder(embedder, func(*embedderConfig) error { return errors.New("option") }); err == nil {
		t.Fatal("embedder option error ignored")
	}
	if _, err := instrumentation.WrapEmbedder(embedder, WithEmbedderMetadata(agentic.ModelMetadata{})); err != nil {
		t.Fatalf("empty provider override: %v", err)
	}
}
