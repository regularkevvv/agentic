// Package otele2e contains the credential-free scenario shared by the
// runnable OpenTelemetry example and the Collector-backed smoke test.
package otele2e

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	agentic "github.com/regularkevvv/agentic"
	agenticotel "github.com/regularkevvv/agentic/otel"

	"go.opentelemetry.io/otel/trace"
)

const (
	// Secret is deliberately placed in prompts, tool arguments, evaluation
	// explanations, and an error. The smoke verifier rejects any raw export.
	Secret   = "e2e-secret-do-not-export"
	Redacted = "[REDACTED]"
)

// Config selects the real Collector endpoints used by Run.
type Config struct {
	Endpoint  string
	HealthURL string
}

// Report is application-level evidence that every deterministic scenario ran.
// The Collector-backed verifier separately proves the telemetry projection.
type Report struct {
	Scenarios        []string
	NestedOutput     string
	SuspendedThenRan bool
	StreamOutput     string
	ProviderError    string
	EmbeddingCount   int
}

// Run executes representative agentic workloads and exports all three OTel
// signals through real OTLP/gRPC clients. It uses no network model or API key.
func Run(ctx context.Context, config Config) (report Report, err error) {
	if err := waitForCollector(ctx, config.HealthURL); err != nil {
		return report, err
	}
	providers, err := newTelemetryProviders(ctx, config.Endpoint)
	if err != nil {
		return report, err
	}
	closed := false
	defer func() {
		if closed {
			return
		}
		shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 15*time.Second)
		defer cancel()
		err = errors.Join(err, providers.shutdown(shutdownCtx))
	}()

	privateInstrumentation, err := agenticotel.New(
		agenticotel.WithTracerProvider(providers.traces),
		agenticotel.WithMeterProvider(providers.metrics),
		agenticotel.WithLoggerProvider(providers.logs),
	)
	if err != nil {
		return report, fmt.Errorf("construct private instrumentation: %w", err)
	}
	contentInstrumentation, err := agenticotel.New(
		agenticotel.WithTracerProvider(providers.traces),
		agenticotel.WithMeterProvider(providers.metrics),
		agenticotel.WithLoggerProvider(providers.logs),
		agenticotel.WithMessageContent(),
		agenticotel.WithToolContent(),
		agenticotel.WithInferenceDetails(),
		agenticotel.WithContentFilter(redactSecret),
		agenticotel.WithMaxContentBytes(64*1024),
	)
	if err != nil {
		return report, fmt.Errorf("construct content instrumentation: %w", err)
	}

	tracer := providers.traces.Tracer("github.com/regularkevvv/agentic/e2e/otel")
	report.Scenarios = []string{
		"nested agent and tool",
		"suspension and resume",
		"streaming",
		"provider error",
		"embeddings",
		"evaluation",
	}
	if report.NestedOutput, err = runNestedScenario(ctx, tracer, contentInstrumentation); err != nil {
		return report, err
	}
	if report.SuspendedThenRan, err = runSuspensionScenario(ctx, tracer, privateInstrumentation); err != nil {
		return report, err
	}
	if report.StreamOutput, err = runStreamingScenario(ctx, tracer, privateInstrumentation); err != nil {
		return report, err
	}
	if report.ProviderError, err = runErrorScenario(ctx, tracer, privateInstrumentation); err != nil {
		return report, err
	}
	if report.EmbeddingCount, err = runEmbeddingScenario(ctx, tracer, privateInstrumentation); err != nil {
		return report, err
	}

	shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 15*time.Second)
	defer cancel()
	if err := providers.shutdown(shutdownCtx); err != nil {
		return report, fmt.Errorf("flush and shut down telemetry: %w", err)
	}
	closed = true
	return report, nil
}

func redactSecret(_ agenticotel.ContentKind, value string) (string, bool) {
	return strings.ReplaceAll(value, Secret, Redacted), true
}

func runNestedScenario(ctx context.Context, tracer trace.Tracer, instrumentation *agenticotel.Instrumentation) (string, error) {
	ctx, scenario := tracer.Start(ctx, "scenario nested-agent-tool")
	defer scenario.End()

	childModel := newScriptedModel("e2e-child-model", func(*agentic.ChatRequest) (*agentic.ChatResponse, error) {
		return textResponse("child-response", "e2e-child-model", "research complete", 3, 2), nil
	})
	child := agentic.NewAgent(
		"Research the delegated question.",
		childModel,
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "researcher", Description: "Nested research agent", Version: "1.0.0"}),
		agentic.WithModelMetadata(modelMetadata()),
	)

	outerModel := newScriptedModel(
		"e2e-outer-model",
		func(request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
			if request == nil || len(request.Tools) != 1 || request.Tools[0].Function.Name != "delegate" {
				return nil, errors.New("outer model did not receive delegate tool")
			}
			return toolResponse("outer-tool", "e2e-outer-model", agentic.ToolUse{
				ID: "delegate-1", Name: "delegate", Input: map[string]any{"query": "investigate " + Secret},
			}), nil
		},
		func(request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
			if !hasSuccessfulToolResult(request, "delegate-1", "delegate") {
				return nil, errors.New("outer model did not receive delegate result")
			}
			return textResponse("outer-response", "e2e-outer-model", "orchestration complete", 5, 3), nil
		},
	)
	outer := agentic.NewAgent(
		"Coordinate work without exposing "+Secret+".",
		outerModel,
		agentic.WithInstrumentation(instrumentation),
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "orchestrator", Description: "Coordinates nested work", Version: "2.0.0"}),
		agentic.WithModelMetadata(modelMetadata()),
	)
	tool, handler, err := agentic.ToolWithContext(
		"delegate",
		"Delegate one research query containing "+Secret+".",
		func(toolCtx context.Context, input delegateInput) (delegateOutput, error) {
			result, err := child.Run(toolCtx, input.Query)
			if err != nil {
				return delegateOutput{}, err
			}
			return delegateOutput{Answer: result.Output}, nil
		},
	)
	if err != nil {
		return "", fmt.Errorf("construct delegate tool: %w", err)
	}
	outer.AddTool(tool, handler)
	result, err := outer.Run(ctx, "coordinate "+Secret,
		agentic.WithRunMetadata(agentic.RunMetadata{ConversationID: "conversation-nested", RunID: "run-nested"}),
	)
	if err != nil {
		return "", fmt.Errorf("run nested scenario: %w", err)
	}
	if result.Output != "orchestration complete" {
		return "", fmt.Errorf("nested output = %q", result.Output)
	}
	score := 1.0
	if err := instrumentation.RecordEvaluation(ctx, agenticotel.EvaluationResult{
		Name:        "correctness",
		ScoreValue:  &score,
		ScoreLabel:  "pass",
		Explanation: "verified while protecting " + Secret,
		ResponseID:  "outer-response",
	}); err != nil {
		return "", fmt.Errorf("record evaluation: %w", err)
	}
	return result.Output, nil
}

func runSuspensionScenario(ctx context.Context, tracer trace.Tracer, instrumentation *agenticotel.Instrumentation) (bool, error) {
	ctx, scenario := tracer.Start(ctx, "scenario suspension-resume")
	defer scenario.End()

	model := newScriptedModel(
		"e2e-approval-model",
		func(*agentic.ChatRequest) (*agentic.ChatResponse, error) {
			return toolResponse("approval-tool", "e2e-approval-model", agentic.ToolUse{
				ID: "approved-1", Name: "approved_action", Input: map[string]any{"value": "execute"},
			}), nil
		},
		func(request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
			if !hasSuccessfulToolResult(request, "approved-1", "approved_action") {
				return nil, errors.New("approval model did not receive resumed result")
			}
			return textResponse("approval-response", "e2e-approval-model", "approved action complete", 4, 2), nil
		},
	)
	agent := agentic.NewAgent(
		"Require approval before executing the action.",
		model,
		agentic.WithInstrumentation(instrumentation),
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "approval-agent"}),
		agentic.WithModelMetadata(modelMetadata()),
	)
	tool, handler, err := agentic.ToolWithContext(
		"approved_action",
		"Execute one approved action.",
		func(context.Context, approvalInput) (map[string]any, error) {
			return map[string]any{"approved": true}, nil
		},
	)
	if err != nil {
		return false, fmt.Errorf("construct approval tool: %w", err)
	}
	agent.AddTool(tool, handler)
	metadata := agentic.RunMetadata{ConversationID: "conversation-approval", RunID: "run-approval"}
	prompt := agentic.NewTextMessage(agentic.RoleUser, "perform the approved action")
	suspended, err := agent.Drive(ctx, agentic.DriveInput{Mode: agentic.DriveStart, Prompt: &prompt},
		agentic.WithRunMetadata(metadata),
		agentic.WithRunToolGate(agentic.ToolGateFunc(func(_ context.Context, calls []agentic.ToolUse) (agentic.ToolBatchDecision, error) {
			dispositions := make([]agentic.ToolDisposition, len(calls))
			for index := range dispositions {
				dispositions[index] = agentic.ToolDisposition{Kind: agentic.ToolDispositionSuspend}
			}
			return agentic.ToolBatchDecision{
				Calls:    dispositions,
				Deferral: &agentic.ToolDeferral{Kind: "human_approval"},
			}, nil
		})),
	)
	if err != nil || suspended == nil || suspended.Status != agentic.ExecutionSuspended || suspended.Suspension == nil || suspended.Result == nil {
		return false, fmt.Errorf("suspend approval scenario: execution=%#v error=%v", suspended, err)
	}
	decisions := make([]agentic.ToolResumeDecision, 0, len(suspended.Result.ToolCalls))
	for _, call := range suspended.Result.ToolCalls {
		decisions = append(decisions, agentic.ToolResumeDecision{CallID: call.ID, Action: agentic.ToolResumeExecute})
	}
	completed, err := agent.Resume(ctx, agentic.ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension, Decisions: decisions,
	}, agentic.WithRunMetadata(metadata))
	if err != nil || completed == nil || completed.Status != agentic.ExecutionCompleted || completed.Result == nil {
		return false, fmt.Errorf("resume approval scenario: execution=%#v error=%v", completed, err)
	}
	if completed.Result.Output != "approved action complete" {
		return false, fmt.Errorf("approval output = %q", completed.Result.Output)
	}
	return true, nil
}

func runStreamingScenario(ctx context.Context, tracer trace.Tracer, instrumentation *agenticotel.Instrumentation) (string, error) {
	ctx, scenario := tracer.Start(ctx, "scenario streaming")
	defer scenario.End()
	agent := agentic.NewAgent(
		"Stream a deterministic answer.",
		streamingModel{},
		agentic.WithInstrumentation(instrumentation),
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "streaming-agent"}),
		agentic.WithModelMetadata(modelMetadata()),
	)
	stream, err := agent.RunStream(ctx, "stream this",
		agentic.WithRunMetadata(agentic.RunMetadata{ConversationID: "conversation-stream", RunID: "run-stream"}),
	)
	if err != nil {
		return "", fmt.Errorf("start streaming scenario: %w", err)
	}
	text, err := stream.Text()
	if err != nil {
		return "", fmt.Errorf("consume streaming scenario: %w", err)
	}
	if text != "hello" {
		return "", fmt.Errorf("stream output = %q", text)
	}
	return text, nil
}

func runErrorScenario(ctx context.Context, tracer trace.Tracer, instrumentation *agenticotel.Instrumentation) (string, error) {
	ctx, scenario := tracer.Start(ctx, "scenario provider-error")
	defer scenario.End()
	want := "provider failed with " + Secret
	agent := agentic.NewAgent(
		"Exercise a provider failure.",
		errorModel{err: errors.New(want)},
		agentic.WithInstrumentation(instrumentation),
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "failing-agent"}),
		agentic.WithModelMetadata(modelMetadata()),
	)
	_, err := agent.Run(ctx, "fail privately",
		agentic.WithRunMetadata(agentic.RunMetadata{ConversationID: "conversation-error", RunID: "run-error"}),
	)
	if err == nil || !strings.Contains(err.Error(), want) {
		return "", fmt.Errorf("provider failure = %v, want text %q", err, want)
	}
	return err.Error(), nil
}

func runEmbeddingScenario(ctx context.Context, tracer trace.Tracer, instrumentation *agenticotel.Instrumentation) (int, error) {
	ctx, scenario := tracer.Start(ctx, "scenario embeddings")
	defer scenario.End()
	embedder, err := instrumentation.WrapEmbedder(
		deterministicEmbedder{},
		agenticotel.WithEmbedderMetadata(agentic.ModelMetadata{
			Provider: "e2e", ServerAddress: "embeddings.e2e.invalid", ServerPort: 443,
		}),
	)
	if err != nil {
		return 0, fmt.Errorf("wrap embedder: %w", err)
	}
	response, err := embedder.Embed(ctx, &agentic.EmbeddingRequest{Input: []string{"alpha", "beta"}})
	if err != nil {
		return 0, fmt.Errorf("run embedding scenario: %w", err)
	}
	if len(response.Vectors) != 2 || len(response.Vectors[0]) != 4 {
		return 0, fmt.Errorf("embedding shape = %dx%d", len(response.Vectors), len(response.Vectors[0]))
	}
	return len(response.Vectors), nil
}

type delegateInput struct {
	Query string `json:"query" description:"Question to delegate"`
}

type delegateOutput struct {
	Answer string `json:"answer"`
}

type approvalInput struct {
	Value string `json:"value" description:"Approved action value"`
}

type scriptedStep func(*agentic.ChatRequest) (*agentic.ChatResponse, error)

type scriptedModel struct {
	name  string
	mu    sync.Mutex
	steps []scriptedStep
	next  int
}

func newScriptedModel(name string, steps ...scriptedStep) *scriptedModel {
	return &scriptedModel{name: name, steps: steps}
}

func (m *scriptedModel) Name() string { return m.name }

func (m *scriptedModel) Request(_ context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.next >= len(m.steps) {
		return nil, fmt.Errorf("%s received unexpected request %d", m.name, m.next+1)
	}
	step := m.steps[m.next]
	m.next++
	return step(request)
}

func modelMetadata() agentic.ModelMetadata {
	return agentic.ModelMetadata{
		Provider: "e2e", Operation: "chat", ServerAddress: "models.e2e.invalid", ServerPort: 443,
	}
}

func textResponse(id, model, text string, inputTokens, outputTokens int) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		ID: id, Model: model,
		Message:         agentic.NewTextMessage(agentic.RoleAssistant, text),
		FinishReason:    agentic.FinishReasonStop,
		RawFinishReason: string(agentic.FinishReasonStop),
		Usage: agentic.Usage{
			PromptTokens: inputTokens, CompletionTokens: outputTokens, TotalTokens: inputTokens + outputTokens,
		},
	}
}

func toolResponse(id, model string, call agentic.ToolUse) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		ID: id, Model: model,
		Message:         agentic.NewToolUseMessage(call),
		FinishReason:    agentic.FinishReasonToolCalls,
		RawFinishReason: string(agentic.FinishReasonToolCalls),
		Usage:           agentic.Usage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
	}
}

func hasSuccessfulToolResult(request *agentic.ChatRequest, id, name string) bool {
	if request == nil {
		return false
	}
	for _, message := range request.Messages {
		for _, result := range message.GetToolResults() {
			if result.ToolUseID == id && result.Name == name && !result.IsError {
				return true
			}
		}
	}
	return false
}

type streamingModel struct{}

func (streamingModel) Name() string { return "e2e-stream-model" }

func (streamingModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return nil, errors.New("streaming model received a non-streaming request")
}

func (streamingModel) RequestStream(ctx context.Context, _ *agentic.ChatRequest) (*agentic.StreamResult, error) {
	events := make(chan agentic.StreamEvent)
	go func() {
		defer close(events)
		sequence := []agentic.StreamEvent{
			{Type: agentic.StreamEventTextDelta, Delta: "hel"},
			{Type: agentic.StreamEventTextDelta, Delta: "lo"},
			{Type: agentic.StreamEventDone, Usage: &agentic.Usage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}, FinishReason: agentic.FinishReasonStop},
		}
		for _, event := range sequence {
			select {
			case <-ctx.Done():
				return
			case <-time.After(5 * time.Millisecond):
			}
			select {
			case <-ctx.Done():
				return
			case events <- event:
			}
		}
	}()
	return agentic.NewStreamResult(events), nil
}

type errorModel struct{ err error }

func (errorModel) Name() string { return "e2e-error-model" }

func (m errorModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return nil, m.err
}

type deterministicEmbedder struct{}

func (deterministicEmbedder) Name() string { return "e2e-embedder" }

func (deterministicEmbedder) Embed(_ context.Context, request *agentic.EmbeddingRequest) (*agentic.EmbeddingResponse, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	vectors := make([][]float32, len(request.Input))
	for index := range request.Input {
		vectors[index] = []float32{float32(index), 0.25, 0.5, 1}
	}
	return &agentic.EmbeddingResponse{
		Vectors: vectors,
		Model:   "e2e-embedder-v1",
		Usage:   agentic.EmbeddingUsage{PromptTokens: 9, TotalTokens: 9},
	}, nil
}
