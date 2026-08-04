package bedrock

import (
	"context"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/internal/core"
)

type mockRuntimeClient struct {
	converseInput  *bedrockruntime.ConverseInput
	converseOutput *bedrockruntime.ConverseOutput
	converseErr    error

	streamInput *bedrockruntime.ConverseStreamInput
	stream      converseStream
	streamErr   error
}

func (m *mockRuntimeClient) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	m.converseInput = params
	return m.converseOutput, m.converseErr
}

func (m *mockRuntimeClient) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (converseStream, error) {
	m.streamInput = params
	return m.stream, m.streamErr
}

type mockConverseStream struct {
	events chan types.ConverseStreamOutput
	err    error
	closed bool
}

func (s *mockConverseStream) Events() <-chan types.ConverseStreamOutput {
	return s.events
}

func (s *mockConverseStream) Close() error {
	s.closed = true
	return nil
}

func (s *mockConverseStream) Err() error {
	return s.err
}

func newMockConverseStream(events ...types.ConverseStreamOutput) *mockConverseStream {
	ch := make(chan types.ConverseStreamOutput, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return &mockConverseStream{events: ch}
}

func TestWithProfileOption(t *testing.T) {
	cfg := &config{}
	WithProfile("dev-profile")(cfg)
	if cfg.profile != "dev-profile" {
		t.Fatalf("expected profile to be set, got %#v", cfg)
	}
}

func TestNewWithInvalidProfileReturnsConfigError(t *testing.T) {
	_, err := New("anthropic.test", WithRegion("us-east-1"), WithProfile("definitely-missing-bedrock-profile"))
	if err == nil {
		t.Fatal("expected config loading error for missing AWS profile")
	}
}

func TestBedrockRequestValidationErrors(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	if _, err := model.Request(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected Request to fail validation")
	}
	if _, err := model.RequestStream(context.Background(), &core.ChatRequest{}); err == nil {
		t.Fatal("expected RequestStream to fail validation")
	}
}

func TestBedrockRequestUsesRuntimeClient(t *testing.T) {
	mock := &mockRuntimeClient{
		converseOutput: &bedrockruntime.ConverseOutput{
			Output: &types.ConverseOutputMemberMessage{
				Value: types.Message{
					Content: []types.ContentBlock{
						&types.ContentBlockMemberText{Value: "answer"},
					},
				},
			},
			StopReason: types.StopReasonEndTurn,
			Usage: &types.TokenUsage{
				InputTokens:  aws.Int32(10),
				OutputTokens: aws.Int32(5),
				TotalTokens:  aws.Int32(15),
			},
		},
	}
	model := &Model{client: mock, modelID: "anthropic.test"}

	resp, err := model.Request(context.Background(), &core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	})
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if mock.converseInput == nil || aws.ToString(mock.converseInput.ModelId) != "anthropic.test" {
		t.Fatalf("unexpected converse input %#v", mock.converseInput)
	}
	if len(mock.converseInput.Messages) != 1 {
		t.Fatalf("expected a single converted message, got %#v", mock.converseInput.Messages)
	}
	if resp.Message.GetTextContent() != "answer" {
		t.Fatalf("unexpected response message %#v", resp.Message)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Fatalf("unexpected usage %#v", resp.Usage)
	}
}

func TestBedrockRequestWrapsRuntimeError(t *testing.T) {
	model := &Model{
		client:  &mockRuntimeClient{converseErr: errors.New("boom")},
		modelID: "anthropic.test",
	}

	_, err := model.Request(context.Background(), &core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	})
	if err == nil || err.Error() != "bedrock: boom" {
		t.Fatalf("expected wrapped runtime error, got %v", err)
	}
}

func TestBuildParamsAndInputs(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}
	temperature := 0.6
	maxTokens := 128
	topP := 0.9
	toolChoice := core.ToolChoiceRequired

	req := &core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "system"),
			{
				Role: core.RoleUser,
				Content: []core.Part{
					{Type: core.ContentText, Text: "hello"},
					agentic.ImageDataPart([]byte("img"), "image/png"),
					{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://example.com/image.png"}},
					{Type: core.ContentDocumentURL, DocumentURL: &core.DocumentURL{URL: "https://example.com/file.pdf"}},
				},
			},
			{
				Role: core.RoleAssistant,
				Content: []core.Part{
					{Type: core.ContentText, Text: "working"},
					{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup", Input: map[string]interface{}{"city": "Lima"}}},
					{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{Text: "reasoning", Signature: "sig"}},
				},
			},
			core.NewToolResultMessage("call_1", `{"temp":72}`, false),
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
		TopP:        &topP,
		Tools: []core.Tool{{
			Type: core.ToolTypeFunction,
			Function: core.Function{
				Name:        "lookup",
				Description: "Lookup weather",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{"type": "string"},
					},
				},
			},
		}},
		ToolChoice: &toolChoice,
		Thinking:   &core.ThinkingConfig{Enabled: true, BudgetTokens: 300},
	}

	params, err := model.buildParams(req)
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if len(params.messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(params.messages))
	}
	if len(params.system) != 1 {
		t.Fatalf("expected 1 system block, got %#v", params.system)
	}
	if params.inferenceConf == nil || params.inferenceConf.Temperature == nil || *params.inferenceConf.Temperature != float32(temperature) {
		t.Fatalf("unexpected inference config %#v", params.inferenceConf)
	}
	if params.toolConfig == nil || len(params.toolConfig.Tools) != 1 {
		t.Fatalf("expected tool config, got %#v", params.toolConfig)
	}
	if params.additionalReq == nil {
		t.Fatal("expected additional model request fields for thinking config")
	}

	input, err := model.buildInput(req)
	if err != nil {
		t.Fatalf("buildInput: %v", err)
	}
	if input.ModelId == nil || *input.ModelId != "anthropic.test" {
		t.Fatalf("unexpected converse input model %#v", input.ModelId)
	}
	streamInput, err := model.buildStreamInput(req)
	if err != nil {
		t.Fatalf("buildStreamInput: %v", err)
	}
	if streamInput.ModelId == nil || *streamInput.ModelId != "anthropic.test" {
		t.Fatalf("unexpected converse stream input model %#v", streamInput.ModelId)
	}
}

func TestConvertOutputMessageIgnoresNonMessageOutput(t *testing.T) {
	msg := convertOutputMessage(nil)
	if msg.Role != core.RoleAssistant || len(msg.Content) != 0 {
		t.Fatalf("expected empty assistant message for non-message output, got %#v", msg)
	}
}

func TestConvertSystemBlocksAndMessage(t *testing.T) {
	if got := convertSystemBlocks("", nil); got != nil {
		t.Fatalf("expected nil system blocks for empty text, got %#v", got)
	}

	// The thinking part here carries no ProviderName, so it degrades to a text
	// block; the cache point now maps to a real Bedrock cachePoint block.
	converted, err := convertMessage(core.Message{
		Role: core.RoleAssistant,
		Content: []core.Part{
			{Type: core.ContentText, Text: "hello"},
			{Type: core.ContentToolUse, ToolUse: &core.ToolUse{ID: "call_1", Name: "lookup", Input: map[string]interface{}{"city": "Lima"}}},
			{Type: core.ContentToolResult, ToolResult: &core.ToolResult{ToolUseID: "call_1", Content: `{"temp":72}`, IsError: true}},
			agentic.ImageDataPart([]byte("img"), "image/png"),
			{Type: core.ContentThinking, Thinking: &core.ThinkingBlock{Text: "reasoning", Signature: "sig"}},
			{Type: core.ContentAudioURL, AudioURL: &core.AudioURL{URL: "https://example.com/audio.mp3"}},
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{}},
		},
	})
	if err != nil {
		t.Fatalf("convertMessage: %v", err)
	}
	if converted == nil || len(converted.Content) != 6 {
		t.Fatalf("expected 6 supported content blocks, got %#v", converted)
	}

	got, err := convertMessage(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentAudioURL, AudioURL: &core.AudioURL{URL: "https://example.com/audio.mp3"}},
			{Type: core.ContentVideoURL, VideoURL: &core.VideoURL{URL: "https://example.com/video.mp4"}},
		},
	})
	if err != nil {
		t.Fatalf("convertMessage: %v", err)
	}
	if got != nil {
		t.Fatalf("expected unsupported-only message to be skipped, got %#v", got)
	}
}

func TestBedrockRequestStreamEmitsEventsAndUsage(t *testing.T) {
	stream := newMockConverseStream(
		&types.ConverseStreamOutputMemberContentBlockStart{
			Value: types.ContentBlockStartEvent{
				ContentBlockIndex: aws.Int32(0),
				Start: &types.ContentBlockStartMemberToolUse{
					Value: types.ToolUseBlockStart{
						ToolUseId: aws.String("call_1"),
						Name:      aws.String("lookup"),
					},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta: &types.ContentBlockDeltaMemberToolUse{
					Value: types.ToolUseBlockDelta{Input: aws.String(`{"city":"Lima"}`)},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(1),
				Delta: &types.ContentBlockDeltaMemberReasoningContent{
					Value: &types.ReasoningContentBlockDeltaMemberText{Value: "thinking"},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(2),
				Delta:             &types.ContentBlockDeltaMemberText{Value: "answer"},
			},
		},
		&types.UnknownUnionMember{Tag: "ignored"},
		&types.ConverseStreamOutputMemberMetadata{
			Value: types.ConverseStreamMetadataEvent{
				Usage: &types.TokenUsage{
					InputTokens:  aws.Int32(10),
					OutputTokens: aws.Int32(5),
					TotalTokens:  aws.Int32(15),
				},
			},
		},
		&types.ConverseStreamOutputMemberMessageStop{
			Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn},
		},
	)
	mock := &mockRuntimeClient{stream: stream}
	model := &Model{client: mock, modelID: "anthropic.test"}

	result, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleUser, "hello"),
		},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range result.Events {
		events = append(events, event)
	}

	if mock.streamInput == nil || aws.ToString(mock.streamInput.ModelId) != "anthropic.test" {
		t.Fatalf("unexpected stream input %#v", mock.streamInput)
	}
	if !stream.closed {
		t.Fatal("expected stream to be closed by RequestStream")
	}
	if len(events) != 5 {
		t.Fatalf("expected 5 emitted events, got %#v", events)
	}
	if events[0].Type != core.StreamEventToolCallStart || events[0].ToolUse == nil || events[0].ToolUse.Name != "lookup" {
		t.Fatalf("unexpected first event %#v", events[0])
	}
	if events[1].Type != core.StreamEventToolCallDelta || events[1].ToolCallID != "call_1" {
		t.Fatalf("unexpected second event %#v", events[1])
	}
	if events[2].Type != core.StreamEventThinkingDelta || events[2].Delta != "thinking" {
		t.Fatalf("unexpected thinking event %#v", events[2])
	}
	if events[3].Type != core.StreamEventTextDelta || events[3].Delta != "answer" {
		t.Fatalf("unexpected text event %#v", events[3])
	}
	if events[4].Type != core.StreamEventDone || events[4].Usage == nil || events[4].Usage.TotalTokens != 15 {
		t.Fatalf("unexpected done event %#v", events[4])
	}
}

func TestBedrockRequestStreamErrors(t *testing.T) {
	t.Run("connect error", func(t *testing.T) {
		model := &Model{
			client:  &mockRuntimeClient{streamErr: errors.New("stream failed")},
			modelID: "anthropic.test",
		}

		_, err := model.RequestStream(context.Background(), &core.ChatRequest{
			Model: "anthropic.test",
			Messages: []core.Message{
				core.NewTextMessage(core.RoleUser, "hello"),
			},
		})
		if err == nil || err.Error() != "bedrock: stream failed" {
			t.Fatalf("expected wrapped stream creation error, got %v", err)
		}
	})

	t.Run("reader error", func(t *testing.T) {
		stream := newMockConverseStream()
		stream.err = errors.New("reader failed")
		model := &Model{
			client:  &mockRuntimeClient{stream: stream},
			modelID: "anthropic.test",
		}

		result, err := model.RequestStream(context.Background(), &core.ChatRequest{
			Model: "anthropic.test",
			Messages: []core.Message{
				core.NewTextMessage(core.RoleUser, "hello"),
			},
		})
		if err != nil {
			t.Fatalf("RequestStream: %v", err)
		}

		if waitErr := result.Wait(); waitErr == nil || waitErr.Error() != "bedrock stream: reader failed" {
			t.Fatalf("expected reader error from stream, got %v", waitErr)
		}
		if !stream.closed {
			t.Fatal("expected errored stream to be closed")
		}
	})
}

func TestConvertToolConfigChoices(t *testing.T) {
	tools := []core.Tool{{
		Type: core.ToolTypeFunction,
		Function: core.Function{
			Name:        "lookup",
			Description: "Lookup weather",
			Parameters:  map[string]interface{}{"type": "object"},
		},
	}}

	none := core.ToolChoiceNone
	if got := convertToolConfig(tools, &none, false); got != nil {
		t.Fatalf("expected nil tool config for none choice, got %#v", got)
	}

	required := core.ToolChoiceRequired
	got := convertToolConfig(tools, &required, false)
	if got == nil {
		t.Fatal("expected required tool config")
	}
	if _, ok := got.ToolChoice.(*types.ToolChoiceMemberAny); !ok {
		t.Fatalf("expected toolChoice.any without thinking, got %T", got.ToolChoice)
	}

	auto := core.ToolChoiceAuto
	if got := convertToolConfig(tools, &auto, false); got == nil || got.ToolChoice == nil {
		t.Fatalf("expected auto tool config, got %#v", got)
	}

	if got := convertToolConfig(tools, nil, false); got == nil || got.ToolChoice != nil {
		t.Fatalf("expected tools with no explicit choice, got %#v", got)
	}
}

// Bedrock rejects toolChoice.any while extended thinking is enabled, so a
// forced choice has to degrade to auto rather than fail the request.
func TestConvertToolConfigDowngradesForcedChoiceUnderThinking(t *testing.T) {
	tools := []core.Tool{{
		Type: core.ToolTypeFunction,
		Function: core.Function{
			Name:       "lookup",
			Parameters: map[string]interface{}{"type": "object"},
		},
	}}

	required := core.ToolChoiceRequired
	got := convertToolConfig(tools, &required, true)
	if got == nil {
		t.Fatal("expected tool config")
	}
	if _, ok := got.ToolChoice.(*types.ToolChoiceMemberAuto); !ok {
		t.Fatalf("expected forced choice to degrade to toolChoice.auto under thinking, got %T", got.ToolChoice)
	}
}

// A forced tool choice reaching buildParams under thinking must arrive at the
// wire as auto, not merely at convertToolConfig.
func TestBuildParamsDowngradesForcedToolChoiceUnderThinking(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}
	required := core.ToolChoiceRequired

	params, err := model.buildParams(&core.ChatRequest{
		Model:      "anthropic.test",
		Messages:   []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		Tools:      []core.Tool{{Type: core.ToolTypeFunction, Function: core.Function{Name: "lookup"}}},
		ToolChoice: &required,
		Thinking:   &core.ThinkingConfig{Enabled: true, BudgetTokens: 1024},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if _, ok := params.toolConfig.ToolChoice.(*types.ToolChoiceMemberAuto); !ok {
		t.Fatalf("expected auto tool choice under thinking, got %T", params.toolConfig.ToolChoice)
	}
}

func TestBedrockConvertResponseAndUsage(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}
	resp := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "answer"},
					&types.ContentBlockMemberToolUse{
						Value: types.ToolUseBlock{
							ToolUseId: aws.String("call_1"),
							Name:      aws.String("lookup"),
							Input:     document.NewLazyDocument(map[string]any{"city": "Lima"}),
						},
					},
					&types.ContentBlockMemberReasoningContent{
						Value: &types.ReasoningContentBlockMemberReasoningText{
							Value: types.ReasoningTextBlock{
								Text:      aws.String("reasoning"),
								Signature: aws.String("sig"),
							},
						},
					},
				},
			},
		},
		StopReason: types.StopReasonToolUse,
		Usage: &types.TokenUsage{
			InputTokens:           aws.Int32(100),
			OutputTokens:          aws.Int32(40),
			TotalTokens:           aws.Int32(140),
			CacheReadInputTokens:  aws.Int32(8),
			CacheWriteInputTokens: aws.Int32(5),
		},
	}

	chatResp := model.convertResponse(resp)
	if chatResp.Model != "anthropic.test" {
		t.Fatalf("unexpected model %q", chatResp.Model)
	}
	if chatResp.FinishReason != core.FinishReasonToolCalls {
		t.Fatalf("expected tool-calls finish reason, got %q", chatResp.FinishReason)
	}
	if chatResp.RawFinishReason != "tool_use" {
		t.Fatalf("expected raw finish reason to be passed through, got %q", chatResp.RawFinishReason)
	}
	msg := chatResp.Message
	if msg.GetTextContent() != "answer" {
		t.Fatalf("unexpected text content %q", msg.GetTextContent())
	}
	if msg.GetThinkingContent() != "reasoning" {
		t.Fatalf("unexpected thinking content %q", msg.GetThinkingContent())
	}
	if len(msg.GetToolUses()) != 1 || msg.GetToolUses()[0].Name != "lookup" {
		t.Fatalf("unexpected tool uses %#v", msg.GetToolUses())
	}
	if chatResp.Usage.PromptTokens != 113 || chatResp.Usage.TotalTokens != 153 || chatResp.Usage.CacheReadTokens != 8 || chatResp.Usage.CacheCreationTokens != 5 {
		t.Fatalf("unexpected usage %#v", chatResp.Usage)
	}
}

// TestConvertStopReasonDefault previously asserted that an unrecognized stop
// reason mapped to FinishReasonStop, which reported a failure as a clean
// success. Anything this library does not know maps to FinishReasonUnknown.
func TestConvertStopReasonDefault(t *testing.T) {
	if got := convertStopReason(types.StopReason("other")); got != core.FinishReasonUnknown {
		t.Fatalf("expected unknown finish reason, got %q", got)
	}
}

// Bedrock takes one top-level system prompt, so every system message in the
// conversation has to be folded in. Keeping only the last one silently dropped
// earlier instructions.
func TestBuildParamsAccumulatesSystemMessages(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	params, err := model.buildParams(&core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "first"),
			core.NewTextMessage(core.RoleUser, "hi"),
			core.NewTextMessage(core.RoleSystem, "second"),
			core.NewTextMessage(core.RoleSystem, "third"),
		},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if len(params.system) != 1 {
		t.Fatalf("expected a single system block, got %#v", params.system)
	}
	text, ok := params.system[0].(*types.SystemContentBlockMemberText)
	if !ok {
		t.Fatalf("expected a text system block, got %T", params.system[0])
	}
	if text.Value != "first\n\nsecond\n\nthird" {
		t.Fatalf("expected all system messages joined, got %q", text.Value)
	}
}

func TestBuildParamsSkipsEmptySystemMessages(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	params, err := model.buildParams(&core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, ""),
			core.NewTextMessage(core.RoleUser, "hi"),
		},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if params.system != nil {
		t.Fatalf("expected no system blocks, got %#v", params.system)
	}
}

// A cache point on a system message caches the whole system prompt.
func TestBuildParamsSystemCachePoint(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	params, err := model.buildParams(&core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			{
				Role: core.RoleSystem,
				Content: []core.Part{
					{Type: core.ContentText, Text: "instructions"},
					{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{TTL: "1h"}},
				},
			},
			core.NewTextMessage(core.RoleUser, "hi"),
		},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if len(params.system) != 2 {
		t.Fatalf("expected text plus cache point system blocks, got %#v", params.system)
	}
	cp, ok := params.system[1].(*types.SystemContentBlockMemberCachePoint)
	if !ok {
		t.Fatalf("expected a system cache point block, got %T", params.system[1])
	}
	if cp.Value.Ttl != types.CacheTTLOneHour {
		t.Fatalf("expected 1h TTL, got %q", cp.Value.Ttl)
	}
}

func TestBuildParamsGeneratedPromptCachePoints(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}
	for _, test := range []struct {
		name      string
		retention core.PromptCacheRetention
		wantTTL   types.CacheTTL
	}{
		{"short", core.PromptCacheShort, types.CacheTTLFiveMinutes},
		{"long", core.PromptCacheLong, types.CacheTTLOneHour},
	} {
		t.Run(test.name, func(t *testing.T) {
			params, err := model.buildParams(&core.ChatRequest{
				Model: "anthropic.test",
				Messages: []core.Message{
					core.NewTextMessage(core.RoleSystem, "instructions"),
					core.NewTextMessage(core.RoleUser, "hi"),
				},
				PromptCache: &core.PromptCacheConfig{Key: "session", Retention: test.retention},
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(params.system) != 2 || len(params.messages) != 1 || len(params.messages[0].Content) != 2 {
				t.Fatalf("params = %#v", params)
			}
			systemPoint := params.system[1].(*types.SystemContentBlockMemberCachePoint)
			messagePoint := params.messages[0].Content[1].(*types.ContentBlockMemberCachePoint)
			if systemPoint.Value.Ttl != test.wantTTL || messagePoint.Value.Ttl != test.wantTTL {
				t.Fatalf("TTLs = %q, %q", systemPoint.Value.Ttl, messagePoint.Value.Ttl)
			}
		})
	}

	explicit, err := model.buildParams(&core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			{Role: core.RoleSystem, Content: []core.Part{{Type: core.ContentText, Text: "system"}, {Type: core.ContentCachePoint, CachePoint: &core.CachePoint{TTL: "1h"}}}},
			{Role: core.RoleUser, Content: []core.Part{{Type: core.ContentText, Text: "hi"}, {Type: core.ContentCachePoint, CachePoint: &core.CachePoint{TTL: "1h"}}}},
		},
		PromptCache: &core.PromptCacheConfig{Key: "session", Retention: core.PromptCacheShort},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(explicit.system) != 2 || len(explicit.messages[0].Content) != 2 {
		t.Fatalf("explicit points duplicated: %#v", explicit)
	}
	appendGeneratedCachePoint(nil, &core.CachePoint{})
	appendGeneratedCachePoint([]types.Message{{}}, &core.CachePoint{})
	appendGeneratedCachePoint([]types.Message{{Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "x"}}}}, nil)
}

// The pinned SDK has ContentBlockMemberCachePoint and multimodal.go documents
// Bedrock cache-point support, so a cache point must reach the wire.
func TestConvertMessageMapsCachePoint(t *testing.T) {
	tests := []struct {
		name string
		ttl  string
		want types.CacheTTL
	}{
		{"five minutes", "5m", types.CacheTTLFiveMinutes},
		{"one hour", "1h", types.CacheTTLOneHour},
		{"unset falls back to service default", "", ""},
		{"unrecognized falls back to service default", "7d", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			converted, err := convertMessage(core.Message{
				Role: core.RoleUser,
				Content: []core.Part{
					{Type: core.ContentText, Text: "cache me"},
					{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{TTL: tt.ttl}},
				},
			})
			if err != nil {
				t.Fatalf("convertMessage: %v", err)
			}
			if converted == nil || len(converted.Content) != 2 {
				t.Fatalf("expected text plus cache point blocks, got %#v", converted)
			}
			cp, ok := converted.Content[1].(*types.ContentBlockMemberCachePoint)
			if !ok {
				t.Fatalf("expected a cache point block, got %T", converted.Content[1])
			}
			if cp.Value.Type != types.CachePointTypeDefault {
				t.Fatalf("expected default cache point type, got %q", cp.Value.Type)
			}
			if cp.Value.Ttl != tt.want {
				t.Fatalf("expected TTL %q, got %q", tt.want, cp.Value.Ttl)
			}
		})
	}
}

// Bedrock rejects a cache point with nothing before it to cache, and rejects
// two in a row, so both are dropped rather than sent.
func TestConvertMessageDropsUnplaceableCachePoints(t *testing.T) {
	converted, err := convertMessage(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{}},
			{Type: core.ContentText, Text: "hi"},
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{}},
			{Type: core.ContentCachePoint, CachePoint: &core.CachePoint{}},
			{Type: core.ContentCachePoint, CachePoint: nil},
		},
	})
	if err != nil {
		t.Fatalf("convertMessage: %v", err)
	}
	if converted == nil || len(converted.Content) != 2 {
		t.Fatalf("expected text plus one cache point, got %#v", converted)
	}
	if _, ok := converted.Content[0].(*types.ContentBlockMemberText); !ok {
		t.Fatalf("expected leading text block, got %T", converted.Content[0])
	}
	if _, ok := converted.Content[1].(*types.ContentBlockMemberCachePoint); !ok {
		t.Fatalf("expected trailing cache point, got %T", converted.Content[1])
	}
}

// Bedrock rejects a reasoning block whose signature it did not issue, so a
// thinking part is only replayed when it is both signed and attributed to this
// provider. Everything else degrades to text instead of failing the request.
func TestConvertMessageThinkingSignatureGuard(t *testing.T) {
	tests := []struct {
		name       string
		thinking   *core.ThinkingBlock
		wantBlocks int
		wantText   string
		wantRedact bool
		wantSigned bool
	}{
		{
			name:       "signed by bedrock replays as reasoning",
			thinking:   &core.ThinkingBlock{Text: "reasoning", Signature: "sig", ProviderName: "bedrock"},
			wantBlocks: 1,
			wantSigned: true,
		},
		{
			name:       "redacted replays as redacted content",
			thinking:   &core.ThinkingBlock{ID: "redacted_thinking", Signature: "cipher", ProviderName: "bedrock"},
			wantBlocks: 1,
			wantRedact: true,
		},
		{
			name:       "another provider degrades to text",
			thinking:   &core.ThinkingBlock{Text: "reasoning", Signature: "sig", ProviderName: "anthropic"},
			wantBlocks: 1,
			wantText:   "reasoning",
		},
		{
			name:       "missing signature degrades to text",
			thinking:   &core.ThinkingBlock{Text: "reasoning", ProviderName: "bedrock"},
			wantBlocks: 1,
			wantText:   "reasoning",
		},
		{
			name:       "unsigned and empty is dropped",
			thinking:   &core.ThinkingBlock{ProviderName: "bedrock"},
			wantBlocks: 0,
		},
		{
			name:       "nil thinking is dropped",
			thinking:   nil,
			wantBlocks: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			converted, err := convertMessage(core.Message{
				Role:    core.RoleAssistant,
				Content: []core.Part{{Type: core.ContentThinking, Thinking: tt.thinking}},
			})
			if err != nil {
				t.Fatalf("convertMessage: %v", err)
			}
			if tt.wantBlocks == 0 {
				if converted != nil {
					t.Fatalf("expected message to be dropped, got %#v", converted)
				}
				return
			}
			if converted == nil || len(converted.Content) != tt.wantBlocks {
				t.Fatalf("expected %d blocks, got %#v", tt.wantBlocks, converted)
			}

			switch {
			case tt.wantSigned:
				rc, ok := converted.Content[0].(*types.ContentBlockMemberReasoningContent)
				if !ok {
					t.Fatalf("expected reasoning content, got %T", converted.Content[0])
				}
				rt, ok := rc.Value.(*types.ReasoningContentBlockMemberReasoningText)
				if !ok {
					t.Fatalf("expected reasoning text, got %T", rc.Value)
				}
				if aws.ToString(rt.Value.Signature) != "sig" {
					t.Fatalf("expected signature to be replayed, got %q", aws.ToString(rt.Value.Signature))
				}
			case tt.wantRedact:
				rc, ok := converted.Content[0].(*types.ContentBlockMemberReasoningContent)
				if !ok {
					t.Fatalf("expected reasoning content, got %T", converted.Content[0])
				}
				red, ok := rc.Value.(*types.ReasoningContentBlockMemberRedactedContent)
				if !ok {
					t.Fatalf("expected redacted content, got %T", rc.Value)
				}
				if string(red.Value) != "cipher" {
					t.Fatalf("expected ciphertext to be replayed, got %q", red.Value)
				}
			default:
				text, ok := converted.Content[0].(*types.ContentBlockMemberText)
				if !ok {
					t.Fatalf("expected degraded text block, got %T", converted.Content[0])
				}
				if text.Value != tt.wantText {
					t.Fatalf("expected degraded text %q, got %q", tt.wantText, text.Value)
				}
			}
		})
	}
}

// A signature arrives as its own delta after the reasoning text and must reach
// the caller, or the turn cannot be replayed to Bedrock.
func TestRequestStreamEmitsThinkingSignatureAndRedactedContent(t *testing.T) {
	stream := newMockConverseStream(
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta: &types.ContentBlockDeltaMemberReasoningContent{
					Value: &types.ReasoningContentBlockDeltaMemberText{Value: "thinking"},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(0),
				Delta: &types.ContentBlockDeltaMemberReasoningContent{
					Value: &types.ReasoningContentBlockDeltaMemberSignature{Value: "sig-abc"},
				},
			},
		},
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(1),
				Delta: &types.ContentBlockDeltaMemberReasoningContent{
					Value: &types.ReasoningContentBlockDeltaMemberRedactedContent{Value: []byte("cipher")},
				},
			},
		},
		&types.ConverseStreamOutputMemberMessageStop{
			Value: types.MessageStopEvent{StopReason: types.StopReasonMaxTokens},
		},
	)
	model := &Model{client: &mockRuntimeClient{stream: stream}, modelID: "anthropic.test"}

	result, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var events []core.StreamEvent
	for event := range result.Events {
		events = append(events, event)
	}
	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %#v", events)
	}

	// Reasoning text carries no signature yet, so it stays unattributed.
	if events[0].Type != core.StreamEventThinkingDelta || events[0].Delta != "thinking" {
		t.Fatalf("unexpected thinking text event %#v", events[0])
	}
	if events[0].Signature != "" || events[0].ProviderName != "" {
		t.Fatalf("expected unsigned thinking text to stay unattributed, got %#v", events[0])
	}

	if events[1].Type != core.StreamEventThinkingDelta || events[1].Signature != "sig-abc" {
		t.Fatalf("expected signature delta, got %#v", events[1])
	}
	if events[1].ProviderName != "bedrock" {
		t.Fatalf("expected signature attributed to bedrock, got %q", events[1].ProviderName)
	}

	if events[2].Type != core.StreamEventThinkingDelta || events[2].Signature != "cipher" {
		t.Fatalf("expected redacted content delta, got %#v", events[2])
	}
	if events[2].ThinkingID != "redacted_thinking" {
		t.Fatalf("expected redacted thinking id, got %q", events[2].ThinkingID)
	}
	if events[2].ProviderName != "bedrock" {
		t.Fatalf("expected redacted content attributed to bedrock, got %q", events[2].ProviderName)
	}

	// The done event reports why generation ended.
	if events[3].Type != core.StreamEventDone || events[3].FinishReason != core.FinishReasonLength {
		t.Fatalf("expected done event carrying length finish reason, got %#v", events[3])
	}
}

// A stream that never reports a stop reason must not claim a clean stop.
func TestRequestStreamWithoutStopReasonIsUnknown(t *testing.T) {
	model := &Model{client: &mockRuntimeClient{stream: newMockConverseStream()}, modelID: "anthropic.test"}

	result, err := model.RequestStream(context.Background(), &core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hello")},
	})
	if err != nil {
		t.Fatalf("RequestStream: %v", err)
	}

	var done core.StreamEvent
	for event := range result.Events {
		if event.Type == core.StreamEventDone {
			done = event
		}
	}
	if done.FinishReason != core.FinishReasonUnknown {
		t.Fatalf("expected unknown finish reason, got %q", done.FinishReason)
	}
}

// ResponseFormat was silently ignored, so a caller asking for JSON got prose
// with no indication the request had been downgraded.
func TestBuildParamsRejectsResponseFormat(t *testing.T) {
	model := &Model{client: &mockRuntimeClient{}, modelID: "anthropic.test"}
	req := &core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		ResponseFormat: &core.ResponseFormat{
			Type: "json_object",
		},
	}

	if _, err := model.buildParams(req); err == nil {
		t.Fatal("expected ResponseFormat to be rejected")
	}
	if _, err := model.Request(context.Background(), req); err == nil {
		t.Fatal("expected Request to reject ResponseFormat")
	}
	if _, err := model.RequestStream(context.Background(), req); err == nil {
		t.Fatal("expected RequestStream to reject ResponseFormat")
	}
}

// Every part of the message here is content Bedrock cannot carry, so the
// message is dropped and the request would reach the API with an empty
// Messages array — a ValidationException with no useful cause.
func TestBuildParamsRejectsEmptyConvertedMessages(t *testing.T) {
	model := &Model{client: &mockRuntimeClient{}, modelID: "anthropic.test"}
	req := &core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			core.NewTextMessage(core.RoleSystem, "system only"),
			{
				Role: core.RoleUser,
				Content: []core.Part{
					{Type: core.ContentAudioURL, AudioURL: &core.AudioURL{URL: "https://example.com/a.mp3"}},
				},
			},
		},
	}

	_, err := model.buildParams(req)
	if err == nil {
		t.Fatal("expected an error when every message is dropped")
	}
	if _, err := model.Request(context.Background(), req); err == nil {
		t.Fatal("expected Request to fail rather than send an empty Messages array")
	}
	if _, err := model.RequestStream(context.Background(), req); err == nil {
		t.Fatal("expected RequestStream to fail rather than send an empty Messages array")
	}
}

// Undecodable image data used to vanish, leaving a prompt that referred to a
// picture the model never received.
func TestConvertMessageRejectsUndecodableImageData(t *testing.T) {
	_, err := convertMessage(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentImageData, ImageData: &core.ImageData{Data: "!!not base64!!", MediaType: "image/png"}},
		},
	})
	if err == nil {
		t.Fatal("expected undecodable image data to fail conversion")
	}
}

func TestBuildParamsWiresStopSequences(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	params, err := model.buildParams(&core.ChatRequest{
		Model:         "anthropic.test",
		Messages:      []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		StopSequences: []string{"STOP", "END"},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if params.inferenceConf == nil {
		t.Fatal("expected inference config for stop sequences")
	}
	if len(params.inferenceConf.StopSequences) != 2 || params.inferenceConf.StopSequences[0] != "STOP" {
		t.Fatalf("unexpected stop sequences %#v", params.inferenceConf.StopSequences)
	}
}

// Converse returns no body-level id, so the AWS request id stands in.
func TestConvertResponseSetsIDFromRequestID(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	out := &bedrockruntime.ConverseOutput{
		Output:     &types.ConverseOutputMemberMessage{Value: types.Message{}},
		StopReason: types.StopReasonEndTurn,
	}
	awsmiddleware.SetRequestIDMetadata(&out.ResultMetadata, "req-abc-123")

	resp := model.convertResponse(out)
	if resp.ID != "req-abc-123" {
		t.Fatalf("expected response id from AWS request id, got %q", resp.ID)
	}
	if resp.RawFinishReason != "end_turn" {
		t.Fatalf("expected raw finish reason, got %q", resp.RawFinishReason)
	}
}

func TestConvertResponseWithoutRequestIDLeavesIDEmpty(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	resp := model.convertResponse(&bedrockruntime.ConverseOutput{
		Output:     &types.ConverseOutputMemberMessage{Value: types.Message{}},
		StopReason: types.StopReasonEndTurn,
	})
	if resp.ID != "" {
		t.Fatalf("expected empty id when AWS reported none, got %q", resp.ID)
	}
}

// A reasoning block is only replayable when signed, so an unsigned one must not
// claim to have come from this provider.
func TestConvertOutputMessageReasoningAttribution(t *testing.T) {
	msg := convertOutputMessage(&types.ConverseOutputMemberMessage{
		Value: types.Message{
			Content: []types.ContentBlock{
				&types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{Text: aws.String("signed"), Signature: aws.String("sig")},
					},
				},
				&types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{Text: aws.String("unsigned")},
					},
				},
				&types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberRedactedContent{Value: []byte("cipher")},
				},
			},
		},
	})
	if len(msg.Content) != 3 {
		t.Fatalf("expected 3 thinking parts, got %#v", msg.Content)
	}
	if msg.Content[0].Thinking.ProviderName != "bedrock" {
		t.Fatalf("expected signed block attributed to bedrock, got %q", msg.Content[0].Thinking.ProviderName)
	}
	if msg.Content[1].Thinking.ProviderName != "" {
		t.Fatalf("expected unsigned block to stay unattributed, got %q", msg.Content[1].Thinking.ProviderName)
	}
	redacted := msg.Content[2].Thinking
	if !redacted.IsRedacted() || redacted.Signature != "cipher" || redacted.ProviderName != "bedrock" {
		t.Fatalf("unexpected redacted thinking block %#v", redacted)
	}
}

func TestOptionsFor(t *testing.T) {
	opts := Options{GuardrailIdentifier: "gr-1"}

	tests := []struct {
		name string
		req  *core.ChatRequest
		want Options
	}{
		{"nil request", nil, Options{}},
		{"no provider options", &core.ChatRequest{}, Options{}},
		{"absent key", &core.ChatRequest{ProviderOptions: map[string]any{"openai": 1}}, Options{}},
		{"value", &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: opts}}, opts},
		{"pointer", &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: &opts}}, opts},
		{"nil pointer", &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: (*Options)(nil)}}, Options{}},
		{"wrong type is ignored", &core.ChatRequest{ProviderOptions: map[string]any{ProviderKey: "nonsense"}}, Options{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := optionsFor(tt.req); got.GuardrailIdentifier != tt.want.GuardrailIdentifier {
				t.Fatalf("optionsFor = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestBuildParamsAppliesProviderOptions(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	params, err := model.buildParams(&core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		ProviderOptions: map[string]any{
			ProviderKey: Options{
				GuardrailIdentifier:          "gr-1",
				GuardrailVersion:             "3",
				PerformanceLatency:           "optimized",
				AdditionalModelRequestFields: map[string]any{"top_k": 40},
			},
		},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}

	cfg := params.guardrailConfig()
	if cfg == nil || aws.ToString(cfg.GuardrailIdentifier) != "gr-1" || aws.ToString(cfg.GuardrailVersion) != "3" {
		t.Fatalf("unexpected guardrail config %#v", cfg)
	}
	streamCfg := params.guardrailStreamConfig()
	if streamCfg == nil || aws.ToString(streamCfg.GuardrailIdentifier) != "gr-1" {
		t.Fatalf("unexpected guardrail stream config %#v", streamCfg)
	}
	if params.performance == nil || params.performance.Latency != types.PerformanceConfigLatencyOptimized {
		t.Fatalf("unexpected performance config %#v", params.performance)
	}
	if params.additionalReq == nil {
		t.Fatal("expected additional model request fields")
	}
}

func TestBuildParamsIgnoresUnusableProviderOptions(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	params, err := model.buildParams(&core.ChatRequest{
		Model:    "anthropic.test",
		Messages: []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		ProviderOptions: map[string]any{
			ProviderKey: Options{PerformanceLatency: "warp-speed"},
		},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if params.performance != nil {
		t.Fatalf("expected unrecognized latency to be ignored, got %#v", params.performance)
	}
	if params.guardrailConfig() != nil || params.guardrailStreamConfig() != nil {
		t.Fatal("expected no guardrail config without an identifier")
	}
	if params.additionalReq != nil {
		t.Fatal("expected no additional request fields")
	}
}

// A guardrail without an explicit version is still sent.
func TestBuildParamsGuardrailWithoutVersion(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	params, err := model.buildParams(&core.ChatRequest{
		Model:           "anthropic.test",
		Messages:        []core.Message{core.NewTextMessage(core.RoleUser, "hi")},
		ProviderOptions: map[string]any{ProviderKey: Options{GuardrailIdentifier: "gr-1"}},
	})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	if cfg := params.guardrailConfig(); cfg == nil || cfg.GuardrailVersion != nil {
		t.Fatalf("unexpected guardrail config %#v", cfg)
	}
	if cfg := params.guardrailStreamConfig(); cfg == nil || cfg.GuardrailVersion != nil {
		t.Fatalf("unexpected guardrail stream config %#v", cfg)
	}
}

// A conversion failure inside a message must abort the request rather than be
// swallowed on the way to the wire.
func TestBuildParamsPropagatesMessageConversionError(t *testing.T) {
	model := &Model{modelID: "anthropic.test"}

	_, err := model.buildParams(&core.ChatRequest{
		Model: "anthropic.test",
		Messages: []core.Message{
			{
				Role: core.RoleUser,
				Content: []core.Part{
					{Type: core.ContentImageData, ImageData: &core.ImageData{Data: "%%%", MediaType: "image/png"}},
				},
			},
		},
	})
	if err == nil {
		t.Fatal("expected the image decode failure to reach the caller")
	}
}
