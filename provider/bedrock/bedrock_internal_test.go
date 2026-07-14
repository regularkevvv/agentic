package bedrock

import (
	"context"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
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
	if resp.Choices[0].Message.GetTextContent() != "answer" {
		t.Fatalf("unexpected response message %#v", resp.Choices[0].Message)
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

	params := model.buildParams(req)
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

	input := model.buildInput(req)
	if input.ModelId == nil || *input.ModelId != "anthropic.test" {
		t.Fatalf("unexpected converse input model %#v", input.ModelId)
	}
	streamInput := model.buildStreamInput(req)
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
	if got := convertSystemBlocks(core.Message{}); got != nil {
		t.Fatalf("expected nil system blocks for empty message, got %#v", got)
	}

	converted := convertMessage(core.Message{
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
	if converted == nil || len(converted.Content) != 5 {
		t.Fatalf("expected 5 supported content blocks, got %#v", converted)
	}

	if got := convertMessage(core.Message{
		Role: core.RoleUser,
		Content: []core.Part{
			{Type: core.ContentAudioURL, AudioURL: &core.AudioURL{URL: "https://example.com/audio.mp3"}},
			{Type: core.ContentVideoURL, VideoURL: &core.VideoURL{URL: "https://example.com/video.mp4"}},
		},
	}); got != nil {
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
	if got := convertToolConfig(tools, &none); got != nil {
		t.Fatalf("expected nil tool config for none choice, got %#v", got)
	}

	required := core.ToolChoiceRequired
	if got := convertToolConfig(tools, &required); got == nil || got.ToolChoice == nil {
		t.Fatalf("expected required tool config, got %#v", got)
	}

	auto := core.ToolChoiceAuto
	if got := convertToolConfig(tools, &auto); got == nil || got.ToolChoice == nil {
		t.Fatalf("expected auto tool config, got %#v", got)
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
	if chatResp.Choices[0].FinishReason != core.FinishReasonToolCalls {
		t.Fatalf("expected tool-calls finish reason, got %q", chatResp.Choices[0].FinishReason)
	}
	msg := chatResp.Choices[0].Message
	if msg.GetTextContent() != "answer" {
		t.Fatalf("unexpected text content %q", msg.GetTextContent())
	}
	if msg.GetThinkingContent() != "reasoning" {
		t.Fatalf("unexpected thinking content %q", msg.GetThinkingContent())
	}
	if len(msg.GetToolUses()) != 1 || msg.GetToolUses()[0].Name != "lookup" {
		t.Fatalf("unexpected tool uses %#v", msg.GetToolUses())
	}
	if chatResp.Usage.CacheReadTokens != 8 || chatResp.Usage.CacheCreationTokens != 5 {
		t.Fatalf("unexpected usage %#v", chatResp.Usage)
	}
}

func TestConvertStopReasonDefault(t *testing.T) {
	if got := convertStopReason(types.StopReason("other")); got != core.FinishReasonStop {
		t.Fatalf("expected default stop finish reason, got %q", got)
	}
}
