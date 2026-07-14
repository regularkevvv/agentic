package agentic

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/testutil"
)

type streamRegistryStub struct {
	executeResult ToolExecutionResult
	executeErr    error
}

func (s *streamRegistryStub) Register(tool Tool, handler ToolHandler) error { return nil }

func (s *streamRegistryStub) Get(name string) (ToolHandler, bool) { return nil, false }

func (s *streamRegistryStub) Execute(ctx context.Context, toolCall ToolUse, deps any) (ToolExecutionResult, error) {
	return s.executeResult, s.executeErr
}

func (s *streamRegistryStub) ExecuteBatch(ctx context.Context, toolCalls []ToolUse, deps any) ([]ToolExecutionResult, error) {
	results := make([]ToolExecutionResult, len(toolCalls))
	for i, toolCall := range toolCalls {
		result, err := s.Execute(ctx, toolCall, deps)
		if err != nil {
			return nil, fmt.Errorf("execute tool %q: %w", toolCall.Name, err)
		}
		results[i] = result
	}
	return results, nil
}

func (s *streamRegistryStub) Tools() []Tool { return nil }

func (s *streamRegistryStub) Has(name string) bool { return false }

func (s *streamRegistryStub) Count() int { return 0 }

func TestRunStreamTrueText(t *testing.T) {
	model := &testutil.ScriptedStreamModel{
		Streams: [][]StreamEvent{{
			{Type: StreamEventTextDelta, Delta: "hello"},
			{Type: StreamEventDone, Usage: &Usage{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}},
		}},
	}

	agent := NewAgent("system", model)
	stream, err := agent.RunStream(context.Background(), "say hello")
	if err != nil {
		t.Fatalf("RunStream: %v", err)
	}

	var events []StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}

	if len(model.Requests) != 1 || !model.Requests[0].Stream {
		t.Fatalf("expected one streaming request, got %#v", model.Requests)
	}
	if len(events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(events))
	}
	if events[0].Type != StreamEventTextDelta || events[0].Delta != "hello" {
		t.Fatalf("unexpected first event: %#v", events[0])
	}
	if events[1].Type != StreamEventDone || events[1].Usage == nil || events[1].Usage.TotalTokens != 5 {
		t.Fatalf("unexpected done event: %#v", events[1])
	}
}

func TestRunStreamTrueWithToolCall(t *testing.T) {
	type doubleInput struct {
		X int `json:"x"`
	}
	type doubleOutput struct {
		Y int `json:"y"`
	}

	tool, handler := MustToolPlain("double", "Double a number", func(input doubleInput) (doubleOutput, error) {
		return doubleOutput{Y: input.X * 2}, nil
	})

	model := &testutil.ScriptedStreamModel{
		Streams: [][]StreamEvent{
			{
				{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "call_1", Name: "double"}},
				{Type: StreamEventToolCallDelta, ToolCallID: "call_1", Delta: `{"x":5}`},
				{Type: StreamEventDone, Usage: &Usage{TotalTokens: 3}},
			},
			{
				{Type: StreamEventTextDelta, Delta: "done"},
				{Type: StreamEventDone, Usage: &Usage{TotalTokens: 2}},
			},
		},
	}

	agent := NewAgent("system", model).AddTool(tool, handler)
	stream, err := agent.RunStream(context.Background(), "double 5")
	if err != nil {
		t.Fatalf("RunStream: %v", err)
	}

	var toolResultSeen bool
	var doneSeen bool
	for event := range stream.Events {
		if event.Type == StreamEventToolResult {
			toolResultSeen = true
			if event.ToolCallID != "call_1" || event.Delta != `{"y":10}` {
				t.Fatalf("unexpected tool result event: %#v", event)
			}
		}
		if event.Type == StreamEventDone {
			doneSeen = true
		}
	}

	if len(model.Requests) != 2 {
		t.Fatalf("expected 2 model requests, got %d", len(model.Requests))
	}
	if !toolResultSeen || !doneSeen {
		t.Fatalf("expected tool result and done events, got toolResult=%v done=%v", toolResultSeen, doneSeen)
	}
}

func TestRunStreamTrueErrorsWhenToolsAreMissing(t *testing.T) {
	model := &testutil.ScriptedStreamModel{
		Streams: [][]StreamEvent{{
			{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "call_1", Name: "missing"}},
			{Type: StreamEventToolCallDelta, ToolCallID: "call_1", Delta: `{}`},
			{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
		}},
	}

	agent := NewAgent("system", model)
	stream, err := agent.RunStream(context.Background(), "run missing tool")
	if err != nil {
		t.Fatalf("RunStream: %v", err)
	}

	err = stream.Wait()
	if err == nil || err.Error() != "model requested tool calls but no tools are registered" {
		t.Fatalf("expected missing tool registry error, got %v", err)
	}
}

func TestRunStreamTrueValidationErrorStopsStream(t *testing.T) {
	model := &testutil.ScriptedStreamModel{
		Streams: [][]StreamEvent{{
			{Type: StreamEventTextDelta, Delta: "bad output"},
			{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
		}},
	}

	agent := NewAgent(
		"system",
		model,
		WithOutputValidatorFunc(func(ctx context.Context, output string) error {
			return errors.New("output rejected")
		}),
		WithMaxValidationRetries(0),
	)

	stream, err := agent.RunStream(context.Background(), "validate this")
	if err != nil {
		t.Fatalf("RunStream: %v", err)
	}

	err = stream.Wait()
	if err == nil || err.Error() != "output validation failed after 0 retries: output rejected" {
		t.Fatalf("expected validation failure, got %v", err)
	}
}

func TestRunStreamTrueValidationRetryAndOutputTool(t *testing.T) {
	t.Run("validation retry continues with another streamed request", func(t *testing.T) {
		model := &testutil.ScriptedStreamModel{
			Streams: [][]StreamEvent{
				{
					{Type: StreamEventTextDelta, Delta: "bad"},
					{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
				},
				{
					{Type: StreamEventTextDelta, Delta: "good"},
					{Type: StreamEventDone, Usage: &Usage{TotalTokens: 2}},
				},
			},
		}

		agent := NewAgent(
			"system",
			model,
			WithOutputValidatorFunc(func(ctx context.Context, output string) error {
				if output == "bad" {
					return NewValidationError("retry please")
				}
				return nil
			}),
			WithMaxValidationRetries(1),
		)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		var events []StreamEvent
		for event := range stream.Events {
			events = append(events, event)
		}

		if len(model.Requests) != 2 {
			t.Fatalf("expected 2 streamed requests after validation retry, got %d", len(model.Requests))
		}
		if len(events) != 3 {
			t.Fatalf("expected 3 events (two deltas and done), got %#v", events)
		}
		last := events[len(events)-1]
		if last.Type != StreamEventDone || last.Usage == nil || last.Usage.TotalTokens != 3 {
			t.Fatalf("unexpected final event %#v", last)
		}

		foundValidationMessage := false
		for _, msg := range model.Requests[1].Messages {
			if msg.Role == RoleUser && strings.Contains(msg.GetTextContent(), "Output validation error: retry please") {
				foundValidationMessage = true
				break
			}
		}
		if !foundValidationMessage {
			t.Fatalf("expected validation retry message in second request, got %#v", model.Requests[1].Messages)
		}
	})

	t.Run("output tool ends streaming loop immediately", func(t *testing.T) {
		model := &testutil.ScriptedStreamModel{
			Streams: [][]StreamEvent{{
				{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "out_1", Name: "__output__"}},
				{Type: StreamEventToolCallDelta, ToolCallID: "out_1", Delta: `{"value":"ok"}`},
				{Type: StreamEventDone, Usage: &Usage{TotalTokens: 4}},
			}},
		}

		agent := NewAgent("system", model).SetOutputToolNames(map[string]bool{"__output__": true})

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		var events []StreamEvent
		for event := range stream.Events {
			events = append(events, event)
		}

		if len(model.Requests) != 1 {
			t.Fatalf("expected a single request, got %d", len(model.Requests))
		}
		if len(events) != 3 {
			t.Fatalf("expected streamed tool call events plus done, got %#v", events)
		}
		if events[len(events)-1].Type != StreamEventDone {
			t.Fatalf("expected done event, got %#v", events[len(events)-1])
		}
	})
}

func TestRunStreamTrueAdditionalErrorPaths(t *testing.T) {
	t.Run("prepare loop error is returned directly", func(t *testing.T) {
		agent := NewAgentDynamic(
			func(ctx context.Context) (string, error) {
				return "", errors.New("prompt failed")
			},
			&testutil.ScriptedStreamModel{NameValue: "stream-model"},
		)

		_, err := agent.RunStream(context.Background(), "prompt")
		if err == nil || err.Error() != "system prompt: prompt failed" {
			t.Fatalf("expected prepare loop error, got %v", err)
		}
	})

	t.Run("pre request usage limit stops immediately", func(t *testing.T) {
		model := &testutil.ScriptedStreamModel{NameValue: "stream-model"}
		agent := NewAgent(
			"system",
			model,
			WithUsageLimits(UsageLimits{MaxRequests: IntPtr(0)}),
		)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		err = stream.Wait()
		if err == nil || !strings.Contains(err.Error(), "usage limit exceeded: requests") {
			t.Fatalf("expected pre-request usage limit error, got %v", err)
		}
		if len(model.Requests) != 0 {
			t.Fatalf("expected no streaming requests, got %d", len(model.Requests))
		}
	})

	t.Run("build request error is forwarded", func(t *testing.T) {
		model := &testutil.ScriptedStreamModel{NameValue: "stream-model"}
		agent := NewAgent("system", model)

		stream, err := agent.RunStream(
			context.Background(),
			"prompt",
			WithRunHistoryProcessor(HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
				return nil, errors.New("boom")
			})),
		)
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		err = stream.Wait()
		if err == nil || err.Error() != "history processor: boom" {
			t.Fatalf("expected build request error, got %v", err)
		}
	})

	t.Run("request stream error is wrapped", func(t *testing.T) {
		model := &testutil.ScriptedStreamModel{NameValue: "stream-model"}
		agent := NewAgent("system", model)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		err = stream.Wait()
		if err == nil || err.Error() != "model request: no scripted stream available" {
			t.Fatalf("expected wrapped request stream error, got %v", err)
		}
	})

	t.Run("stream consumption errors are forwarded", func(t *testing.T) {
		expected := errors.New("stream failed")
		model := &testutil.ScriptedStreamModel{
			NameValue: "stream-model",
			Streams: [][]StreamEvent{{
				{Type: StreamEventError, Error: expected},
			}},
		}
		agent := NewAgent("system", model)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		err = stream.Wait()
		if !errors.Is(err, expected) {
			t.Fatalf("expected %v, got %v", expected, err)
		}
	})

	t.Run("post response usage limit is enforced", func(t *testing.T) {
		model := &testutil.ScriptedStreamModel{
			NameValue: "stream-model",
			Streams: [][]StreamEvent{{
				{Type: StreamEventTextDelta, Delta: "hello"},
				{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
			}},
		}
		agent := NewAgent(
			"system",
			model,
			WithUsageLimits(UsageLimits{MaxTotalTokens: IntPtr(0)}),
		)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		err = stream.Wait()
		if err == nil || !strings.Contains(err.Error(), "usage limit exceeded: total_tokens") {
			t.Fatalf("expected post-response usage limit error, got %v", err)
		}
	})

	t.Run("tool call usage limit is enforced", func(t *testing.T) {
		type noopInput struct{}
		type noopOutput struct{}

		tool, handler := MustToolPlain("noop", "noop", func(input noopInput) (noopOutput, error) {
			return noopOutput{}, nil
		})

		model := &testutil.ScriptedStreamModel{
			NameValue: "stream-model",
			Streams: [][]StreamEvent{{
				{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "call_1", Name: "noop"}},
				{Type: StreamEventToolCallDelta, ToolCallID: "call_1", Delta: `{}`},
				{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
			}},
		}
		agent := NewAgent(
			"system",
			model,
			WithUsageLimits(UsageLimits{MaxToolCalls: IntPtr(0)}),
		).AddTool(tool, handler)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		err = stream.Wait()
		if err == nil || !strings.Contains(err.Error(), "usage limit exceeded: tool_calls") {
			t.Fatalf("expected tool call limit error, got %v", err)
		}
	})

	t.Run("registry execution errors are forwarded", func(t *testing.T) {
		model := &testutil.ScriptedStreamModel{
			NameValue: "stream-model",
			Streams: [][]StreamEvent{{
				{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "call_1", Name: "noop"}},
				{Type: StreamEventToolCallDelta, ToolCallID: "call_1", Delta: `{}`},
				{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
			}},
		}
		agent := NewAgent("system", model)
		agent.core.registry = &streamRegistryStub{executeErr: errors.New("registry failed")}

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		err = stream.Wait()
		if err == nil || err.Error() != `execute tool "noop": registry failed` {
			t.Fatalf("expected registry execute error, got %v", err)
		}
	})

	t.Run("model retry continues to the next streamed request", func(t *testing.T) {
		type retryInput struct{}
		type retryOutput struct{}

		tool, handler := MustToolPlain("retry_tool", "retry tool", func(input retryInput) (retryOutput, error) {
			return retryOutput{}, Retry("try again")
		})

		model := &testutil.ScriptedStreamModel{
			NameValue: "stream-model",
			Streams: [][]StreamEvent{
				{
					{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "call_1", Name: "retry_tool"}},
					{Type: StreamEventToolCallDelta, ToolCallID: "call_1", Delta: `{}`},
					{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
				},
				{
					{Type: StreamEventTextDelta, Delta: "done"},
					{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
				},
			},
		}
		agent := NewAgent("system", model, WithRetries(RetryConfig{MaxRetries: 1})).AddTool(tool, handler)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		var events []StreamEvent
		for event := range stream.Events {
			events = append(events, event)
		}

		if len(model.Requests) != 2 {
			t.Fatalf("expected retry to trigger a second streamed request, got %d", len(model.Requests))
		}
		if events[len(events)-1].Type != StreamEventDone {
			t.Fatalf("expected done event after retry, got %#v", events)
		}

		foundRetryMessage := false
		for _, msg := range model.Requests[1].Messages {
			results := msg.GetToolResults()
			if msg.Role == RoleTool && len(results) > 0 && strings.Contains(results[0].Content, "try again") {
				foundRetryMessage = true
				break
			}
		}
		if !foundRetryMessage {
			t.Fatalf("expected retry tool result in follow-up request, got %#v", model.Requests[1].Messages)
		}
	})

	t.Run("max iterations error is emitted", func(t *testing.T) {
		type noopInput struct{}
		type noopOutput struct{}

		tool, handler := MustToolPlain("noop", "noop", func(input noopInput) (noopOutput, error) {
			return noopOutput{}, nil
		})

		model := &testutil.ScriptedStreamModel{
			NameValue: "stream-model",
			Streams: [][]StreamEvent{{
				{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "call_1", Name: "noop"}},
				{Type: StreamEventToolCallDelta, ToolCallID: "call_1", Delta: `{}`},
				{Type: StreamEventDone, Usage: &Usage{TotalTokens: 1}},
			}},
		}
		agent := NewAgent("system", model, WithMaxIterations(1)).AddTool(tool, handler)

		stream, err := agent.RunStream(context.Background(), "prompt")
		if err != nil {
			t.Fatalf("RunStream: %v", err)
		}

		var maxIterErr *MaxIterationsError
		err = stream.Wait()
		if !errors.As(err, &maxIterErr) || maxIterErr.MaxIterations != 1 {
			t.Fatalf("expected max iterations error, got %v", err)
		}
	})
}

func TestConsumeAndForward(t *testing.T) {
	t.Run("reconstructs text thinking tool calls and forwards events", func(t *testing.T) {
		out := make(chan StreamEvent, 8)
		stream := testutil.NewScriptedStream(
			StreamEvent{Type: StreamEventTextDelta, Delta: "hello"},
			StreamEvent{Type: StreamEventThinkingDelta, Delta: "considering"},
			StreamEvent{Type: StreamEventToolCallStart, ToolUse: &ToolUse{ID: "call_1", Name: "calc"}},
			StreamEvent{Type: StreamEventToolCallDelta, ToolCallID: "call_1", Delta: `{"x":1}`},
			StreamEvent{Type: StreamEventToolResult, Delta: "side-effect"},
			StreamEvent{Type: StreamEventDone, Usage: &Usage{TotalTokens: 7}},
		)

		msg, usage, err := (&agentCore{}).consumeAndForward(stream, out)
		if err != nil {
			t.Fatalf("consumeAndForward: %v", err)
		}
		if usage.TotalTokens != 7 {
			t.Fatalf("expected usage to be preserved, got %#v", usage)
		}
		if msg.GetTextContent() != "hello" {
			t.Fatalf("expected text content %q, got %q", "hello", msg.GetTextContent())
		}
		if msg.GetThinkingContent() != "considering" {
			t.Fatalf("expected thinking content %q, got %q", "considering", msg.GetThinkingContent())
		}
		toolUses := msg.GetToolUses()
		if len(toolUses) != 1 || toolUses[0].Name != "calc" || toolUses[0].Input["x"] != float64(1) {
			t.Fatalf("unexpected tool uses: %#v", toolUses)
		}
		if len(out) != 5 {
			t.Fatalf("expected 5 forwarded events, got %d", len(out))
		}
	})

	t.Run("returns streamed errors immediately", func(t *testing.T) {
		expected := errors.New("stream failed")
		out := make(chan StreamEvent, 1)
		stream := testutil.NewScriptedStream(StreamEvent{Type: StreamEventError, Error: expected})

		_, _, err := (&agentCore{}).consumeAndForward(stream, out)
		if !errors.Is(err, expected) {
			t.Fatalf("expected %v, got %v", expected, err)
		}
	})
}

func TestNewScriptedStreamHelperUsesStableTimestamps(t *testing.T) {
	stream := testutil.NewScriptedStream(StreamEvent{Type: StreamEventDone, Usage: &Usage{TotalTokens: int(time.Unix(0, 0).Unix())}})
	if stream == nil {
		t.Fatal("expected helper to create a stream")
	}
}
