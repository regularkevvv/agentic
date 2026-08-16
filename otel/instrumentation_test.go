package agenticotel_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/regularkevvv/agentic"
	agenticotel "github.com/regularkevvv/agentic/otel"
	providertest "github.com/regularkevvv/agentic/provider/test"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"
)

type telemetryFixture struct {
	instrumentation *agenticotel.Instrumentation
	spans           *tracetest.SpanRecorder
	metrics         *metric.ManualReader
	logs            *logExporter
}

func newTelemetryFixture(t *testing.T, options ...agenticotel.Option) telemetryFixture {
	t.Helper()
	spans := tracetest.NewSpanRecorder()
	tracerProvider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spans))
	reader := metric.NewManualReader()
	meterProvider := metric.NewMeterProvider(metric.WithReader(reader))
	logs := &logExporter{}
	loggerProvider := log.NewLoggerProvider(log.WithProcessor(log.NewSimpleProcessor(logs)))
	options = append([]agenticotel.Option{
		agenticotel.WithTracerProvider(tracerProvider),
		agenticotel.WithMeterProvider(meterProvider),
		agenticotel.WithLoggerProvider(loggerProvider),
	}, options...)
	instrumentation, err := agenticotel.New(options...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() {
		_ = loggerProvider.Shutdown(context.Background())
		_ = meterProvider.Shutdown(context.Background())
		_ = tracerProvider.Shutdown(context.Background())
	})
	return telemetryFixture{instrumentation: instrumentation, spans: spans, metrics: reader, logs: logs}
}

type logExporter struct {
	mu      sync.Mutex
	records []log.Record
}

func (e *logExporter) Export(_ context.Context, records []log.Record) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	for index := range records {
		e.records = append(e.records, records[index].Clone())
	}
	return nil
}
func (*logExporter) Shutdown(context.Context) error   { return nil }
func (*logExporter) ForceFlush(context.Context) error { return nil }
func (e *logExporter) snapshot() []log.Record {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]log.Record(nil), e.records...)
}

type handler struct{ name string }

func (h handler) Name() string { return h.name }
func (h handler) Execute(context.Context, map[string]any, any) (any, error) {
	return map[string]any{"temperature": 21}, nil
}

func TestAgentTelemetryTreeMetricsAndPrivacyDefault(t *testing.T) {
	fixture := newTelemetryFixture(t)
	model := providertest.NewTestModel(
		providertest.ModelResponse{ToolCalls: []agentic.ToolUse{{ID: "call-1", Name: "weather", Input: map[string]any{"city": "Paris"}}}},
		providertest.ModelResponse{Text: "sunny"},
	)
	registry := agentic.NewRegistry()
	if err := registry.Register(agentic.Tool{
		Type:     agentic.ToolTypeFunction,
		Function: agentic.Function{Name: "weather", Description: "private description", Parameters: map[string]any{"type": "object"}},
	}, handler{name: "weather"}); err != nil {
		t.Fatal(err)
	}
	agent := agentic.NewAgent("private system", model,
		agentic.WithInstrumentation(fixture.instrumentation),
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "planner", Description: "Plans work", Version: "1.2.3"}),
		agentic.WithModelMetadata(agentic.ModelMetadata{Provider: "openai", Operation: "chat", ServerAddress: "api.example", ServerPort: 8443}),
	).SetRegistry(registry)
	result, err := agent.Run(context.Background(), "private prompt", agentic.WithRunMetadata(agentic.RunMetadata{ConversationID: "conversation-1", RunID: "run-1"}))
	if err != nil || result.Output != "sunny" {
		t.Fatalf("Run = %#v, %v", result, err)
	}

	ended := fixture.spans.Ended()
	if len(ended) != 4 {
		t.Fatalf("ended spans = %d, want 4", len(ended))
	}
	byName := spansByName(ended)
	agentSpan := byName["invoke_agent planner"]
	if agentSpan == nil || agentSpan.SpanKind() != oteltrace.SpanKindInternal {
		t.Fatalf("agent span = %#v", agentSpan)
	}
	for _, name := range []string{"chat test:mock", "execute_tool weather"} {
		spans := byName[name]
		if spans == nil || spans.Parent().SpanID() != agentSpan.SpanContext().SpanID() {
			t.Fatalf("%s parent is not invoke_agent", name)
		}
	}
	modelSpans := spansNamed(ended, "chat test:mock")
	if len(modelSpans) != 2 || modelSpans[0].SpanKind() != oteltrace.SpanKindClient {
		t.Fatalf("model spans = %#v", modelSpans)
	}
	attrs := attributesByKey(modelSpans[0].Attributes())
	if attrs["gen_ai.operation.name"].AsString() != "chat" || attrs["gen_ai.provider.name"].AsString() != "openai" || attrs["server.address"].AsString() != "api.example" {
		t.Fatalf("model attributes = %#v", attrs)
	}
	if attributesByKey(agentSpan.Attributes())["gen_ai.conversation.id"].AsString() != "conversation-1" {
		t.Fatalf("agent correlation attributes = %#v", agentSpan.Attributes())
	}
	agentSpanAttrs := attributesByKey(agentSpan.Attributes())
	if agentSpanAttrs["gen_ai.agent.description"].AsString() != "Plans work" {
		t.Fatalf("agent description attributes = %#v", agentSpanAttrs)
	}
	if _, exists := agentSpanAttrs["gen_ai.agent.version"]; exists {
		t.Fatal("in-process invoke_agent span included gen_ai.agent.version")
	}
	if agentSpanAttrs["agentic.agent.version"].AsString() != "1.2.3" {
		t.Fatalf("agentic agent version attributes = %#v", agentSpanAttrs)
	}
	for _, span := range ended {
		for _, key := range []string{"gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions", "gen_ai.tool.definitions", "gen_ai.tool.call.arguments", "gen_ai.tool.call.result"} {
			if _, exists := attributesByKey(span.Attributes())[key]; exists {
				t.Fatalf("default telemetry leaked %s on %s", key, span.Name())
			}
		}
	}
	if records := fixture.logs.snapshot(); len(records) != 0 {
		t.Fatalf("successful default run logs = %d, want 0", len(records))
	}

	metrics := collectMetrics(t, fixture.metrics)
	for _, name := range []string{
		"gen_ai.client.token.usage", "gen_ai.client.operation.duration",
		"gen_ai.invoke_agent.duration", "gen_ai.invoke_agent.inference_calls",
		"gen_ai.invoke_agent.tool_calls", "gen_ai.execute_tool.duration",
	} {
		if metrics[name].Name == "" {
			t.Errorf("missing metric %s", name)
		}
	}
	if got := histogramCount[int64](t, metrics["gen_ai.invoke_agent.inference_calls"]); got != 1 {
		t.Fatalf("inference histogram points = %d, want 1", got)
	}
	if value := histogramSum[int64](t, metrics["gen_ai.invoke_agent.inference_calls"]); value != 2 {
		t.Fatalf("inference calls = %d, want 2", value)
	}
	if value := histogramSum[int64](t, metrics["gen_ai.invoke_agent.tool_calls"]); value != 1 {
		t.Fatalf("tool calls = %d, want 1", value)
	}
	assertHistogramBounds[float64](t, metrics["gen_ai.client.operation.duration"], []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92})
	assertHistogramBounds[int64](t, metrics["gen_ai.client.token.usage"], []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864})
	assertHistogramBounds[float64](t, metrics["gen_ai.invoke_agent.duration"], []float64{0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6})
	assertHistogramBounds[int64](t, metrics["gen_ai.invoke_agent.inference_calls"], []float64{1, 2, 4, 8, 16, 32, 64, 128})
	assertHistogramBounds[float64](t, metrics["gen_ai.execute_tool.duration"], []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92})
}

func TestFinishReasonAndFailedToolResultProjectionFollowSchemas(t *testing.T) {
	fixture := newTelemetryFixture(t, agenticotel.WithMessageContent(), agenticotel.WithToolContent(), agenticotel.WithInferenceDetails())

	temperature, topP, maxTokens := 0.25, 0.9, 128
	_, invocation := fixture.instrumentation.StartAgent(context.Background(), agentic.AgentOperation{
		ModelName: "agent-model",
		Request: agentic.ChatRequest{
			Temperature: &temperature,
			TopP:        &topP,
			MaxTokens:   &maxTokens,
			Stream:      true,
			ResponseFormat: &agentic.ResponseFormat{
				Type: "json_schema",
			},
			Tools: []agentic.Tool{{
				Type: agentic.ToolTypeFunction,
				Function: agentic.Function{
					Name:       "lookup",
					Parameters: map[string]any{"type": "object"},
				},
			}},
		},
	})
	invocation.End(agentic.AgentOperationResult{
		Status:       agentic.ExecutionSuspended,
		FinishReason: agentic.FinishReasonToolCalls,
	})
	_, failedTool := fixture.instrumentation.StartTool(context.Background(), agentic.ToolOperation{
		Call: agentic.ToolUse{ID: "failed-1", Name: "failed"},
	})
	failedTool.End(agentic.ToolOperationResult{Result: agentic.ToolExecutionResult{
		Content: "private failure payload",
		IsError: true,
	}})
	_, successfulTool := fixture.instrumentation.StartTool(context.Background(), agentic.ToolOperation{
		Call: agentic.ToolUse{ID: "ok-1", Name: "ok"}, Attempt: 2, HandlerResumed: true,
	})
	successfulTool.End(agentic.ToolOperationResult{Result: agentic.ToolExecutionResult{
		Content: map[string]any{"answer": 42},
	}})
	_, suspendedTool := fixture.instrumentation.StartTool(context.Background(), agentic.ToolOperation{
		Call: agentic.ToolUse{ID: "suspended-1", Name: "suspended"},
	})
	suspendedTool.End(agentic.ToolOperationResult{
		Result:    agentic.ToolExecutionResult{Content: map[string]any{"private": "checkpoint"}},
		Suspended: true,
	})
	_, failedModel := fixture.instrumentation.StartModelRequest(context.Background(), agentic.ModelOperation{
		Model:   agentic.ModelMetadata{Provider: "custom", Operation: "chat"},
		Request: agentic.ChatRequest{Model: "failed-model"},
	})
	failedModel.End(agentic.ModelOperationResult{Response: &agentic.ChatResponse{
		Model:        "failed-model",
		FinishReason: agentic.FinishReasonError,
		Usage:        agentic.Usage{PromptTokens: 3, CompletionTokens: 1},
	}})
	prior := agentic.NewTextMessage(agentic.RoleAssistant, "prior response")
	_, noOutput := fixture.instrumentation.StartAgent(context.Background(), agentic.AgentOperation{
		Agent: agentic.AgentIdentity{Name: "no-output"},
		Input: []agentic.Message{prior},
	})
	noOutput.End(agentic.AgentOperationResult{
		Status:       agentic.ExecutionFailed,
		Messages:     []agentic.Message{prior},
		FinishReason: agentic.FinishReasonError,
	})

	ended := fixture.spans.Ended()
	invocationAttrs := attributesByKey(spansByName(ended)["invoke_agent"].Attributes())
	if got := invocationAttrs["gen_ai.response.finish_reasons"].AsStringSlice(); !slices.Equal(got, []string{"tool_calls"}) {
		t.Fatalf("response finish reasons = %#v, want provider-facing tool_calls", got)
	}
	if invocationAttrs["gen_ai.request.max_tokens"].AsInt64() != 128 ||
		invocationAttrs["gen_ai.request.temperature"].AsFloat64() != 0.25 ||
		invocationAttrs["gen_ai.request.top_p"].AsFloat64() != 0.9 ||
		invocationAttrs["gen_ai.output.type"].AsString() != "json" ||
		invocationAttrs["gen_ai.tool.definitions"].Type() != attribute.SLICE {
		t.Fatalf("invoke-agent request attributes = %#v", invocationAttrs)
	}
	if _, exists := invocationAttrs["gen_ai.request.stream"]; exists {
		t.Fatal("invoke-agent span included inference-only gen_ai.request.stream")
	}
	failedAttrs := attributesByKey(spansByName(ended)["execute_tool failed"].Attributes())
	if _, exists := failedAttrs["gen_ai.tool.call.result"]; exists {
		t.Fatal("failed tool span reported an execution result")
	}
	if failedAttrs["error.type"].AsString() != "tool_error" {
		t.Fatalf("failed tool error.type = %#v", failedAttrs)
	}
	successAttrs := attributesByKey(spansByName(ended)["execute_tool ok"].Attributes())
	if successAttrs["gen_ai.tool.call.result"].Type() != attribute.MAP {
		t.Fatalf("successful tool result = %#v", successAttrs)
	}
	if successAttrs["agentic.tool.attempt"].AsInt64() != 2 || !successAttrs["agentic.tool.handler_resumed"].AsBool() {
		t.Fatalf("Agentic tool lifecycle attributes = %#v", successAttrs)
	}
	suspendedAttrs := attributesByKey(spansByName(ended)["execute_tool suspended"].Attributes())
	if suspendedAttrs["agentic.tool.outcome"].AsString() != "suspended" {
		t.Fatalf("suspended tool lifecycle attributes = %#v", suspendedAttrs)
	}
	if _, exists := suspendedAttrs["gen_ai.tool.call.result"]; exists {
		t.Fatal("suspended tool span reported an execution result")
	}
	modelSpan := spansByName(ended)["chat failed-model"]
	modelAttrs := attributesByKey(modelSpan.Attributes())
	if modelSpan.Status().Code != codes.Error || modelAttrs["error.type"].AsString() != "provider_error" {
		t.Fatalf("in-band model failure status/attributes = %#v / %#v", modelSpan.Status(), modelAttrs)
	}
	metrics := collectMetrics(t, fixture.metrics)
	if !histogramHasAttribute[float64](t, metrics["gen_ai.client.operation.duration"], "error.type", "provider_error") {
		t.Fatal("in-band model failure duration has no provider_error dimension")
	}
	if histogramHasAttribute[int64](t, metrics["gen_ai.client.token.usage"], "error.type", "provider_error") {
		t.Fatal("token usage metric included an undocumented error.type dimension")
	}
	records := fixture.logs.snapshot()
	if len(records) != 2 || records[0].EventName() != "gen_ai.client.operation.exception" ||
		logAttributes(records[0])["exception.type"].AsString() != "provider_error" ||
		records[1].EventName() != "gen_ai.client.inference.operation.details" ||
		logAttributes(records[1])["error.type"].AsString() != "provider_error" {
		t.Fatalf("in-band failure inference details = %#v", records)
	}
	if _, exists := attributesByKey(spansByName(ended)["invoke_agent no-output"].Attributes())["gen_ai.output.messages"]; exists {
		t.Fatal("agent failure with no new response repeated an input message as output")
	}
}

func TestHandoffChildInheritsInstrumentationAndCorrelation(t *testing.T) {
	fixture := newTelemetryFixture(t)
	child := agentic.NewAgent("child system",
		providertest.NewTestModel(providertest.ModelResponse{Text: "child answer"}),
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "child"}),
	)
	parent := agentic.NewAgent("parent system",
		providertest.NewTestModel(
			providertest.ModelResponse{ToolCalls: []agentic.ToolUse{{ID: "handoff-1", Name: "delegate", Input: map[string]any{"task": "help"}}}},
			providertest.ModelResponse{Text: "parent answer"},
		),
		agentic.WithInstrumentation(fixture.instrumentation),
		agentic.WithAgentIdentity(agentic.AgentIdentity{Name: "parent"}),
	).AddHandoff(agentic.NewHandoff("delegate", "delegate work", child))
	metadata := agentic.RunMetadata{ConversationID: "conversation", RunID: "run"}
	if _, err := parent.Run(context.Background(), "start", agentic.WithRunMetadata(metadata)); err != nil {
		t.Fatal(err)
	}

	ended := fixture.spans.Ended()
	parentAgent := spansByName(ended)["invoke_agent parent"]
	childAgent := spansByName(ended)["invoke_agent child"]
	handoff := spansByName(ended)["execute_tool delegate"]
	childModels := spansNamed(ended, "chat test:mock")
	if parentAgent == nil || childAgent == nil || handoff == nil || len(childModels) != 3 {
		t.Fatalf("handoff trace names = %#v", spanNames(ended))
	}
	var childModel sdktrace.ReadOnlySpan
	for _, model := range childModels {
		if model.Parent().SpanID() == childAgent.SpanContext().SpanID() {
			childModel = model
			break
		}
	}
	if childModel == nil || handoff.Parent().SpanID() != parentAgent.SpanContext().SpanID() ||
		childAgent.Parent().SpanID() != handoff.SpanContext().SpanID() {
		t.Fatalf("handoff ancestry is not parent agent -> tool -> child agent -> child model")
	}
	childAttrs := attributesByKey(childAgent.Attributes())
	if childAttrs["gen_ai.conversation.id"].AsString() != metadata.ConversationID || childAttrs["agentic.run.id"].AsString() != metadata.RunID {
		t.Fatalf("child correlation attributes = %#v", childAttrs)
	}
}

func TestOptInContentInferenceLogsAndEvaluationAreStructured(t *testing.T) {
	fixture := newTelemetryFixture(t,
		agenticotel.WithMessageContent(),
		agenticotel.WithToolContent(),
		agenticotel.WithInferenceDetails(),
		agenticotel.WithContentFilter(func(kind agenticotel.ContentKind, value string) (string, bool) {
			if kind == agenticotel.ContentMessageText && value == "secret" {
				return "[redacted]", true
			}
			return value, true
		}),
	)
	model := providertest.NewTestModel(providertest.ModelResponse{Text: "answer"})
	agent := agentic.NewAgent("system", model, agentic.WithInstrumentation(fixture.instrumentation))
	if _, err := agent.Run(context.Background(), "secret"); err != nil {
		t.Fatal(err)
	}

	modelSpan := spansNamed(fixture.spans.Ended(), "chat test:mock")[0]
	spanAttrs := attributesByKey(modelSpan.Attributes())
	if spanAttrs["gen_ai.input.messages"].Type() != attribute.SLICE || spanAttrs["gen_ai.system_instructions"].Type() != attribute.SLICE || spanAttrs["gen_ai.output.messages"].Type() != attribute.SLICE {
		t.Fatalf("content attributes are not structured: %#v", spanAttrs)
	}
	if got := spanAttrs["gen_ai.input.messages"].String(); !contains(got, "[redacted]") || contains(got, "secret") {
		t.Fatalf("filtered messages = %s", got)
	}
	records := fixture.logs.snapshot()
	if len(records) != 1 || records[0].EventName() != "gen_ai.client.inference.operation.details" || records[0].TraceID() != modelSpan.SpanContext().TraceID() {
		t.Fatalf("inference records = %#v", records)
	}
	logAttrs := logAttributes(records[0])
	if logAttrs["gen_ai.input.messages"].Type() != attribute.SLICE {
		t.Fatalf("log input messages = %#v", logAttrs)
	}
	score := 0.0
	ctx := oteltrace.ContextWithSpanContext(context.Background(), modelSpan.SpanContext())
	if err := fixture.instrumentation.RecordEvaluation(ctx, agenticotel.EvaluationResult{Name: "correctness", ScoreValue: &score, ScoreLabel: "pass", Explanation: "verified", ResponseID: "r-1"}); err != nil {
		t.Fatal(err)
	}
	records = fixture.logs.snapshot()
	if len(records) != 2 || records[1].EventName() != "gen_ai.evaluation.result" || records[1].TraceID() != modelSpan.SpanContext().TraceID() {
		t.Fatalf("evaluation record = %#v", records)
	}
}

type errorModel struct{ err error }

func (m errorModel) Name() string { return "broken" }
func (m errorModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return nil, m.err
}

func TestExceptionLogIsCorrelatedAndPrivateByDefault(t *testing.T) {
	fixture := newTelemetryFixture(t)
	secret := errors.New("provider response contains secret")
	agent := agentic.NewAgent("", errorModel{err: secret}, agentic.WithInstrumentation(fixture.instrumentation))
	if _, err := agent.Run(context.Background(), "prompt"); !errors.Is(err, secret) {
		t.Fatalf("Run error = %v", err)
	}
	modelSpan := spansNamed(fixture.spans.Ended(), "chat broken")[0]
	if modelSpan.Status().Code != codes.Error || len(modelSpan.Events()) != 0 {
		t.Fatalf("model error span status/events = %#v / %#v", modelSpan.Status(), modelSpan.Events())
	}
	records := fixture.logs.snapshot()
	if len(records) != 1 || records[0].EventName() != "gen_ai.client.operation.exception" || records[0].Severity() != 13 || records[0].TraceID() != modelSpan.SpanContext().TraceID() {
		t.Fatalf("exception record = %#v", records)
	}
	attrs := logAttributes(records[0])
	if attrs["exception.type"].AsString() == "" {
		t.Fatalf("exception.type absent: %#v", attrs)
	}
	if _, leaked := attrs["exception.message"]; leaked {
		t.Fatal("default exception log leaked message")
	}
}

func TestStreamingChunkMetricsAndEmbeddingWrapper(t *testing.T) {
	fixture := newTelemetryFixture(t)
	streamModel := &testStreamModel{}
	agent := agentic.NewAgent("", streamModel,
		agentic.WithInstrumentation(fixture.instrumentation),
		agentic.WithModelMetadata(agentic.ModelMetadata{Provider: "custom_stream", Operation: "chat"}),
	)
	stream, err := agent.RunStream(context.Background(), "prompt")
	if err != nil {
		t.Fatal(err)
	}
	if text, err := stream.Text(); err != nil || text != "hello" {
		t.Fatalf("stream Text = %q, %v", text, err)
	}
	metrics := collectMetrics(t, fixture.metrics)
	if histogramCount[float64](t, metrics["gen_ai.client.operation.time_to_first_chunk"]) != 1 {
		t.Fatal("TTFC metric was not recorded exactly once")
	}
	if histogramCount[float64](t, metrics["gen_ai.client.operation.time_per_output_chunk"]) != 1 {
		t.Fatal("per-output-chunk metric was not recorded after the first chunk")
	}
	streamingBounds := []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}
	assertHistogramBounds[float64](t, metrics["gen_ai.client.operation.time_to_first_chunk"], streamingBounds)
	assertHistogramBounds[float64](t, metrics["gen_ai.client.operation.time_per_output_chunk"], streamingBounds)
	modelSpan := spansNamed(fixture.spans.Ended(), "chat stream-model")[0]
	if _, ok := attributesByKey(modelSpan.Attributes())["gen_ai.response.time_to_first_chunk"]; !ok {
		t.Fatal("streaming model span has no TTFC attribute")
	}

	embedFixture := newTelemetryFixture(t)
	embedder, err := embedFixture.instrumentation.WrapEmbedder(
		providertest.NewTestEmbedder(3),
		agenticotel.WithEmbedderMetadata(agentic.ModelMetadata{Provider: "cohere", ServerAddress: "embed.example"}),
	)
	if err != nil {
		t.Fatal(err)
	}
	response, err := embedder.Embed(context.Background(), &agentic.EmbeddingRequest{Input: []string{"one", "two"}})
	if err != nil || len(response.Vectors) != 2 {
		t.Fatalf("Embed = %#v, %v", response, err)
	}
	embedSpan := spansNamed(embedFixture.spans.Ended(), "embeddings test:embedder")[0]
	embedAttrs := attributesByKey(embedSpan.Attributes())
	if embedAttrs["gen_ai.operation.name"].AsString() != "embeddings" || embedAttrs["gen_ai.provider.name"].AsString() != "cohere" || embedAttrs["gen_ai.usage.input_tokens"].AsInt64() == 0 || embedAttrs["gen_ai.embeddings.dimension.count"].AsInt64() != 3 {
		t.Fatalf("embedding attributes = %#v", embedAttrs)
	}
	embedMetrics := collectMetrics(t, embedFixture.metrics)
	if histogramCount[float64](t, embedMetrics["gen_ai.client.operation.duration"]) != 1 || histogramCount[int64](t, embedMetrics["gen_ai.client.token.usage"]) != 1 {
		t.Fatalf("embedding metrics = %#v", embedMetrics)
	}
}

type testStreamModel struct{}

func (testStreamModel) Name() string { return "stream-model" }
func (testStreamModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return nil, errors.New("non-streaming request")
}
func (testStreamModel) RequestStream(context.Context, *agentic.ChatRequest) (*agentic.StreamResult, error) {
	ch := make(chan agentic.StreamEvent, 3)
	ch <- agentic.StreamEvent{Type: agentic.StreamEventTextDelta, Delta: "hel"}
	ch <- agentic.StreamEvent{Type: agentic.StreamEventTextDelta, Delta: "lo"}
	ch <- agentic.StreamEvent{Type: agentic.StreamEventDone, Usage: &agentic.Usage{PromptTokens: 2, CompletionTokens: 1}, FinishReason: agentic.FinishReasonStop}
	close(ch)
	return agentic.NewStreamResult(ch), nil
}

func TestConfigurationExceptionContentAndContentLimits(t *testing.T) {
	if _, err := agenticotel.New(nil); err == nil {
		t.Fatal("nil option accepted")
	}
	if _, err := agenticotel.New(agenticotel.WithMaxContentBytes(-1)); err == nil {
		t.Fatal("negative content limit accepted")
	}
	fixture := newTelemetryFixture(t,
		agenticotel.WithMessageContent(),
		agenticotel.WithExceptionContent(),
		agenticotel.WithMaxContentBytes(2),
	)
	secret := errors.New("visible only after opt in")
	agent := agentic.NewAgent("system", errorModel{err: secret}, agentic.WithInstrumentation(fixture.instrumentation))
	_, _ = agent.Run(context.Background(), "message too large")
	modelSpan := spansNamed(fixture.spans.Ended(), "chat broken")[0]
	if _, exists := attributesByKey(modelSpan.Attributes())["gen_ai.input.messages"]; exists {
		t.Fatal("oversize content was not omitted")
	}
	if len(modelSpan.Events()) != 1 {
		t.Fatalf("opt-in exception event count = %d", len(modelSpan.Events()))
	}
	records := fixture.logs.snapshot()
	if len(records) != 1 || !contains(logAttributes(records[0])["exception.message"].AsString(), secret.Error()) {
		t.Fatalf("opt-in exception record = %#v", records)
	}
	if err := fixture.instrumentation.RecordEvaluation(context.Background(), agenticotel.EvaluationResult{}); err == nil {
		t.Fatal("empty evaluation name accepted")
	}
	if _, err := fixture.instrumentation.WrapEmbedder(nil); err == nil {
		t.Fatal("nil embedder accepted")
	}
}

func spansByName(spans []sdktrace.ReadOnlySpan) map[string]sdktrace.ReadOnlySpan {
	result := make(map[string]sdktrace.ReadOnlySpan)
	for _, span := range spans {
		result[span.Name()] = span
	}
	return result
}

func spansNamed(spans []sdktrace.ReadOnlySpan, name string) []sdktrace.ReadOnlySpan {
	var result []sdktrace.ReadOnlySpan
	for _, span := range spans {
		if span.Name() == name {
			result = append(result, span)
		}
	}
	return result
}

func spanNames(spans []sdktrace.ReadOnlySpan) []string {
	names := make([]string, 0, len(spans))
	for _, span := range spans {
		names = append(names, span.Name())
	}
	return names
}

func attributesByKey(attrs []attribute.KeyValue) map[string]attribute.Value {
	result := make(map[string]attribute.Value, len(attrs))
	for _, attr := range attrs {
		result[string(attr.Key)] = attr.Value
	}
	return result
}

func logAttributes(record log.Record) map[string]attribute.Value {
	result := make(map[string]attribute.Value)
	record.WalkAttributes(func(attr attribute.KeyValue) bool { result[string(attr.Key)] = attr.Value; return true })
	return result
}

func collectMetrics(t *testing.T, reader *metric.ManualReader) map[string]metricdata.Metrics {
	t.Helper()
	var resource metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &resource); err != nil {
		t.Fatal(err)
	}
	result := make(map[string]metricdata.Metrics)
	for _, scope := range resource.ScopeMetrics {
		for _, measurement := range scope.Metrics {
			result[measurement.Name] = measurement
		}
	}
	return result
}

func histogramCount[N int64 | float64](t *testing.T, measurement metricdata.Metrics) uint64 {
	t.Helper()
	histogram, ok := measurement.Data.(metricdata.Histogram[N])
	if !ok {
		t.Fatalf("%s data = %T", measurement.Name, measurement.Data)
	}
	var total uint64
	for _, point := range histogram.DataPoints {
		total += point.Count
	}
	return total
}

func histogramSum[N int64 | float64](t *testing.T, measurement metricdata.Metrics) N {
	t.Helper()
	histogram, ok := measurement.Data.(metricdata.Histogram[N])
	if !ok {
		t.Fatalf("%s data = %T", measurement.Name, measurement.Data)
	}
	var total N
	for _, point := range histogram.DataPoints {
		total += point.Sum
	}
	return total
}

func histogramHasAttribute[N int64 | float64](t *testing.T, measurement metricdata.Metrics, key, value string) bool {
	t.Helper()
	histogram, ok := measurement.Data.(metricdata.Histogram[N])
	if !ok {
		t.Fatalf("%s data = %T", measurement.Name, measurement.Data)
	}
	for _, point := range histogram.DataPoints {
		for _, attr := range point.Attributes.ToSlice() {
			if string(attr.Key) == key && attr.Value.AsString() == value {
				return true
			}
		}
	}
	return false
}

func assertHistogramBounds[N int64 | float64](t *testing.T, measurement metricdata.Metrics, want []float64) {
	t.Helper()
	histogram, ok := measurement.Data.(metricdata.Histogram[N])
	if !ok || len(histogram.DataPoints) == 0 {
		t.Fatalf("%s data = %T with %d points", measurement.Name, measurement.Data, len(histogram.DataPoints))
	}
	if got := histogram.DataPoints[0].Bounds; !slices.Equal(got, want) {
		t.Fatalf("%s bounds = %v, want %v", measurement.Name, got, want)
	}
}

func contains(value, target string) bool {
	for index := 0; index+len(target) <= len(value); index++ {
		if value[index:index+len(target)] == target {
			return true
		}
	}
	return false
}
