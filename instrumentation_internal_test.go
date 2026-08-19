package agentic

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic/internal/testutil"
)

type instrumentationContextKey string

type nilReturningInstrumentation struct{}

func (nilReturningInstrumentation) StartAgent(context.Context, AgentOperation) (context.Context, AgentOperationSpan) {
	return nil, nil
}
func (nilReturningInstrumentation) StartModelRequest(context.Context, ModelOperation) (context.Context, ModelOperationSpan) {
	return nil, nil
}
func (nilReturningInstrumentation) StartTool(context.Context, ToolOperation) (context.Context, ToolOperationSpan) {
	return nil, nil
}

type recordingInstrumentation struct {
	mu           sync.Mutex
	events       []string
	agents       []AgentOperation
	models       []ModelOperation
	tools        []ToolOperation
	streamEvents []StreamEvent
	agentResults []AgentOperationResult
	modelResults []ModelOperationResult
	toolResults  []ToolOperationResult
	panicAt      string
}

func (r *recordingInstrumentation) add(event string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *recordingInstrumentation) StartAgent(ctx context.Context, operation AgentOperation) (context.Context, AgentOperationSpan) {
	if r.panicAt == "start_agent" {
		panic("observer")
	}
	r.mu.Lock()
	r.agents = append(r.agents, cloneAgentOperation(operation))
	r.events = append(r.events, "agent:start")
	r.mu.Unlock()
	// Mutating the observation copy must not alter execution.
	operation.Input[0].Content[0].Text = "observer mutation"
	if operation.Request.Temperature != nil {
		*operation.Request.Temperature = 2
	}
	if len(operation.Request.Tools) > 0 {
		operation.Request.Tools[0].Function.Name = "observer mutation"
	}
	if operation.Request.PromptCache != nil {
		operation.Request.PromptCache.Key = "observer mutation"
	}
	return context.WithValue(ctx, instrumentationContextKey("agent"), true), recordingAgentSpan{r}
}

func (r *recordingInstrumentation) StartModelRequest(ctx context.Context, operation ModelOperation) (context.Context, ModelOperationSpan) {
	if r.panicAt == "start_model" {
		panic("observer")
	}
	if ctx.Value(instrumentationContextKey("agent")) != true {
		panic("missing agent context")
	}
	r.mu.Lock()
	r.models = append(r.models, cloneModelOperation(operation))
	r.events = append(r.events, "model:start")
	r.mu.Unlock()
	operation.Request.Messages[0].Content[0].Text = "observer mutation"
	if operation.Request.Temperature != nil {
		*operation.Request.Temperature = 2
	}
	return context.WithValue(ctx, instrumentationContextKey("model"), true), recordingModelSpan{r}
}

func (r *recordingInstrumentation) StartTool(ctx context.Context, operation ToolOperation) (context.Context, ToolOperationSpan) {
	if r.panicAt == "start_tool" {
		panic("observer")
	}
	if ctx.Value(instrumentationContextKey("agent")) != true {
		panic("missing agent context")
	}
	r.mu.Lock()
	r.tools = append(r.tools, cloneToolOperation(operation))
	r.events = append(r.events, "tool:start:"+operation.Call.Name)
	r.mu.Unlock()
	operation.Call.Input["mutated"] = true
	return context.WithValue(ctx, instrumentationContextKey("tool"), operation.Call.Name), recordingToolSpan{r}
}

type recordingAgentSpan struct{ recorder *recordingInstrumentation }

func (s recordingAgentSpan) End(result AgentOperationResult) {
	defer s.recorder.add("agent:end")
	if s.recorder.panicAt == "end_agent" {
		panic("observer")
	}
	s.recorder.mu.Lock()
	s.recorder.agentResults = append(s.recorder.agentResults, cloneAgentOperationResult(result))
	s.recorder.mu.Unlock()
	result.Messages[0].Content[0].Text = "observer mutation"
}

type recordingModelSpan struct{ recorder *recordingInstrumentation }

func (s recordingModelSpan) ObserveStreamEvent(event StreamEvent) {
	if s.recorder.panicAt == "stream" {
		panic("observer")
	}
	s.recorder.mu.Lock()
	s.recorder.streamEvents = append(s.recorder.streamEvents, event)
	s.recorder.mu.Unlock()
}
func (s recordingModelSpan) End(result ModelOperationResult) {
	if s.recorder.panicAt == "end_model" {
		panic("observer")
	}
	s.recorder.mu.Lock()
	s.recorder.modelResults = append(s.recorder.modelResults, cloneModelOperationResult(result))
	s.recorder.events = append(s.recorder.events, "model:end")
	s.recorder.mu.Unlock()
	if result.Response != nil {
		result.Response.Message.Content[0].Text = "observer mutation"
	}
}

type recordingToolSpan struct{ recorder *recordingInstrumentation }

func (s recordingToolSpan) End(result ToolOperationResult) {
	if s.recorder.panicAt == "end_tool" {
		panic("observer")
	}
	s.recorder.mu.Lock()
	s.recorder.toolResults = append(s.recorder.toolResults, cloneToolOperationResult(result))
	s.recorder.events = append(s.recorder.events, "tool:end:"+result.Result.ToolName)
	s.recorder.mu.Unlock()
}

type instrumentationModel struct {
	mu        sync.Mutex
	calls     int
	contexts  []context.Context
	temps     []float64
	cacheKeys []string
}

func (m *instrumentationModel) Name() string { return "instrumented-model" }
func (m *instrumentationModel) ModelMetadata() ModelMetadata {
	return ModelMetadata{Provider: "example", Operation: "chat", ServerAddress: "models.example"}
}
func (m *instrumentationModel) Request(ctx context.Context, request *ChatRequest) (*ChatResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.contexts = append(m.contexts, ctx)
	if request.Temperature != nil {
		m.temps = append(m.temps, *request.Temperature)
	}
	if request.PromptCache != nil {
		m.cacheKeys = append(m.cacheKeys, request.PromptCache.Key)
	}
	m.calls++
	if m.calls == 1 {
		return &ChatResponse{Model: m.Name(), Message: NewToolUseMessage(
			ToolUse{ID: "one", Name: "one", Input: map[string]any{"value": 1}},
			ToolUse{ID: "two", Name: "two", Input: map[string]any{"value": 2}},
		), FinishReason: FinishReasonToolCalls, Usage: Usage{PromptTokens: 3, CompletionTokens: 2}}, nil
	}
	return &ChatResponse{ID: "response", Model: m.Name(), Message: NewTextMessage(RoleAssistant, "done"), FinishReason: FinishReasonStop, Usage: Usage{PromptTokens: 4, CompletionTokens: 1}}, nil
}

type instrumentationHandler struct {
	name string
	seen chan context.Context
}

func (h instrumentationHandler) Name() string { return h.name }
func (h instrumentationHandler) Execute(ctx context.Context, input map[string]any, _ any) (any, error) {
	if input["mutated"] != nil {
		return nil, errors.New("observer mutated handler input")
	}
	h.seen <- ctx
	return map[string]any{"ok": true}, nil
}

func TestInstrumentationLifecycleContextAndDefensiveCopies(t *testing.T) {
	recorder := &recordingInstrumentation{}
	model := &instrumentationModel{}
	registry := NewRegistry()
	seen := make(chan context.Context, 2)
	for _, name := range []string{"one", "two"} {
		definition := Tool{Type: ToolTypeFunction, Function: Function{Name: name, Parameters: map[string]any{"type": "object"}}}
		if err := registry.Register(definition, instrumentationHandler{name: name, seen: seen}); err != nil {
			t.Fatal(err)
		}
	}
	agent := NewAgent("system", model,
		WithInstrumentation(recorder),
		WithAgentIdentity(AgentIdentity{Name: "planner", Version: "v1"}),
		WithTemperature(0.4),
		WithPromptCache(PromptCacheConfig{Key: "cache", Retention: PromptCacheShort}),
	).SetRegistry(registry)
	result, err := agent.Run(context.Background(), "original", WithRunMetadata(RunMetadata{ConversationID: "session", RunID: "run"}))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Output != "done" || result.Messages[1].GetTextContent() != "original" {
		t.Fatalf("observer changed execution result: %#v", result)
	}
	for range 2 {
		ctx := <-seen
		if ctx.Value(instrumentationContextKey("agent")) != true || ctx.Value(instrumentationContextKey("tool")) == nil {
			t.Fatalf("tool context did not contain observer ancestry")
		}
	}
	if len(model.contexts) != 2 || model.contexts[0].Value(instrumentationContextKey("model")) != true {
		t.Fatalf("model did not receive model child context")
	}
	if len(model.temps) != 2 || model.temps[0] != 0.4 || model.temps[1] != 0.4 {
		t.Fatalf("observer changed model request settings: %#v", model.temps)
	}
	if len(model.cacheKeys) != 2 || model.cacheKeys[0] != "cache" || model.cacheKeys[1] != "cache" {
		t.Fatalf("observer changed prompt cache settings: %#v", model.cacheKeys)
	}
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	if len(recorder.agents) != 1 || recorder.agents[0].Mode != AgentInvocationStart || recorder.agents[0].Run.ConversationID != "session" || recorder.agents[0].ModelName != model.Name() {
		t.Fatalf("agent operations = %#v", recorder.agents)
	}
	if recorder.agents[0].Request.Temperature == nil || *recorder.agents[0].Request.Temperature != 0.4 ||
		len(recorder.agents[0].Request.Tools) != 2 || recorder.agents[0].Request.PromptCache == nil ||
		recorder.agents[0].Request.PromptCache.Key != "cache" {
		t.Fatalf("agent invocation request settings = %#v", recorder.agents[0].Request)
	}
	if recorder.agents[0].Model.Provider != "example" || len(recorder.models) != 2 || len(recorder.tools) != 2 {
		t.Fatalf("model/tool operations = %#v / %#v", recorder.models, recorder.tools)
	}
	if len(recorder.agentResults) != 1 || recorder.agentResults[0].Status != ExecutionCompleted || recorder.agentResults[0].Usage.Requests != 2 {
		t.Fatalf("agent results = %#v", recorder.agentResults)
	}
	if len(recorder.modelResults) != 2 || len(recorder.toolResults) != 2 {
		t.Fatalf("terminal callbacks = %d model, %d tool", len(recorder.modelResults), len(recorder.toolResults))
	}
	if recorder.events[0] != "agent:start" || recorder.events[len(recorder.events)-1] != "agent:end" {
		t.Fatalf("event boundary = %#v", recorder.events)
	}
}

func TestInstrumentationClosesSuspensionAndResumeAsDistinctInvocations(t *testing.T) {
	recorder := &recordingInstrumentation{}
	model := &instrumentationModel{}
	registry := NewRegistry()
	seen := make(chan context.Context, 2)
	for _, name := range []string{"one", "two"} {
		definition := Tool{Type: ToolTypeFunction, Function: Function{Name: name, Parameters: map[string]any{"type": "object"}}}
		if err := registry.Register(definition, instrumentationHandler{name: name, seen: seen}); err != nil {
			t.Fatal(err)
		}
	}
	agent := NewAgent("", model,
		WithInstrumentation(recorder),
		WithAgentIdentity(AgentIdentity{Name: "resumable"}),
	).SetRegistry(registry)
	prompt := NewTextMessage(RoleUser, "run")
	metadata := RunMetadata{ConversationID: "conversation", RunID: "durable-run"}
	suspended, err := agent.Drive(context.Background(), DriveInput{Mode: DriveStart, Prompt: &prompt},
		WithRunMetadata(metadata),
		WithRunToolGate(ToolGateFunc(func(_ context.Context, calls []ToolUse) (ToolBatchDecision, error) {
			dispositions := make([]ToolDisposition, len(calls))
			for index := range dispositions {
				dispositions[index] = ToolDisposition{Kind: ToolDispositionSuspend}
			}
			return ToolBatchDecision{Calls: dispositions, Deferral: &ToolDeferral{Kind: "approval"}}, nil
		})),
	)
	if err != nil || suspended.Status != ExecutionSuspended || suspended.Suspension == nil {
		t.Fatalf("suspended = %#v, %v", suspended, err)
	}

	decisions := make([]ToolResumeDecision, 0, len(suspended.Result.ToolCalls))
	for _, call := range suspended.Result.ToolCalls {
		decisions = append(decisions, ToolResumeDecision{CallID: call.ID, Action: ToolResumeExecute})
	}
	completed, err := agent.Resume(context.Background(), ResumeInput{
		History: suspended.Result.Messages, Suspension: *suspended.Suspension, Decisions: decisions,
	}, WithRunMetadata(metadata))
	if err != nil || completed.Status != ExecutionCompleted {
		t.Fatalf("resumed = %#v, %v", completed, err)
	}

	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	if len(recorder.agents) != 2 || recorder.agents[0].Mode != AgentInvocationStart || recorder.agents[1].Mode != AgentInvocationResume {
		t.Fatalf("invocations = %#v", recorder.agents)
	}
	if len(recorder.agentResults) != 2 || recorder.agentResults[0].Status != ExecutionSuspended || recorder.agentResults[1].Status != ExecutionCompleted {
		t.Fatalf("invocation results = %#v", recorder.agentResults)
	}
	if recorder.agents[0].Run != metadata || recorder.agents[1].Run != metadata {
		t.Fatalf("run correlation = %#v", recorder.agents)
	}
	if len(recorder.tools) != 2 || recorder.tools[0].HandlerResumed || recorder.tools[1].HandlerResumed {
		t.Fatalf("approval-gated tools were marked as resumed handlers: %#v", recorder.tools)
	}
}

func TestInstrumentationStreamingEventsAndPanicsNeverChangeExecution(t *testing.T) {
	recorder := &recordingInstrumentation{}
	model := &testutil.ScriptedStreamModel{Streams: [][]StreamEvent{{
		{Type: StreamEventTextDelta, Delta: "hel"},
		{Type: StreamEventTextDelta, Delta: "lo"},
		{Type: StreamEventDone, Usage: &Usage{PromptTokens: 1, CompletionTokens: 1}, FinishReason: FinishReasonStop},
	}}}
	agent := NewAgent("", model, WithInstrumentation(recorder))
	stream, err := agent.RunStream(context.Background(), "prompt")
	if err != nil {
		t.Fatal(err)
	}
	text, err := stream.Text()
	if err != nil || text != "hello" {
		t.Fatalf("Text = %q, %v", text, err)
	}
	if len(recorder.streamEvents) != 3 {
		t.Fatalf("stream observations = %d", len(recorder.streamEvents))
	}
	if len(recorder.agents) != 1 || !recorder.agents[0].Request.Stream {
		t.Fatalf("streaming invocation request = %#v", recorder.agents)
	}

	for _, panicAt := range []string{"start_agent", "start_model", "end_model", "end_agent"} {
		t.Run(panicAt, func(t *testing.T) {
			model := &instrumentationModel{calls: 1}
			agent := NewAgent("", model, WithInstrumentation(&recordingInstrumentation{panicAt: panicAt}))
			result, err := agent.Run(context.Background(), "safe")
			if err != nil || result.Output != "done" {
				t.Fatalf("telemetry panic changed run: %#v, %v", result, err)
			}
		})
	}
}

func TestInstrumentationNilReturnsNoopsAndRunOverrides(t *testing.T) {
	ctx := context.WithValue(context.Background(), instrumentationContextKey("original"), true)
	agentCtx, agentSpan := safeStartAgent(ctx, nilReturningInstrumentation{}, AgentOperation{})
	modelCtx, modelSpan := safeStartModel(ctx, nilReturningInstrumentation{}, ModelOperation{})
	toolCtx, toolSpan := safeStartTool(ctx, nilReturningInstrumentation{}, ToolOperation{})
	if agentCtx != ctx || modelCtx != ctx || toolCtx != ctx {
		t.Fatal("nil observer context replaced the execution context")
	}
	agentSpan.End(AgentOperationResult{})
	modelSpan.ObserveStreamEvent(StreamEvent{})
	modelSpan.End(ModelOperationResult{})
	toolSpan.End(ToolOperationResult{})

	_, noopAgent := safeStartAgent(ctx, nil, AgentOperation{})
	_, noopModel := safeStartModel(ctx, nil, ModelOperation{})
	_, noopTool := safeStartTool(ctx, nil, ToolOperation{})
	noopAgent.End(AgentOperationResult{})
	noopModel.ObserveStreamEvent(StreamEvent{})
	noopModel.End(ModelOperationResult{})
	noopTool.End(ToolOperationResult{})

	panicking := &recordingInstrumentation{panicAt: "stream"}
	safeObserveStreamEvent(recordingModelSpan{panicking}, StreamEvent{Type: StreamEventTextDelta})
	panicking.panicAt = "end_tool"
	safeEndTool(recordingToolSpan{panicking}, ToolOperationResult{})

	agentLevel := &recordingInstrumentation{}
	runLevel := &recordingInstrumentation{}
	agent := NewAgent("", &instrumentationModel{calls: 1},
		WithInstrumentation(agentLevel),
		WithModelMetadata(ModelMetadata{}),
	)
	result, err := agent.Run(context.Background(), "override", WithRunInstrumentation(runLevel))
	if err != nil || result.Output != "done" || len(agentLevel.agents) != 0 || len(runLevel.agents) != 1 {
		t.Fatalf("run observer override result=%#v err=%v agent=%d run=%d", result, err, len(agentLevel.agents), len(runLevel.agents))
	}
	if runLevel.agents[0].Model.Provider != "custom" || runLevel.agents[0].Model.Operation != "chat" {
		t.Fatalf("empty model override was not normalized: %#v", runLevel.agents[0].Model)
	}
	if _, err := agent.Run(context.Background(), "disabled", WithRunInstrumentation(nil)); err != nil {
		t.Fatal(err)
	}
	if len(agentLevel.agents) != 0 || len(runLevel.agents) != 1 {
		t.Fatal("nil run override did not disable observation")
	}
}
