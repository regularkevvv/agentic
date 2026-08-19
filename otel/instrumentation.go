// Package agenticotel instruments Agentic with OpenTelemetry GenAI semantic
// conventions while keeping the core framework free of OpenTelemetry imports.
package agenticotel

import (
	"context"
	"errors"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/regularkevvv/agentic"

	otelapi "go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/log"
	logglobal "go.opentelemetry.io/otel/log/global"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

const (
	InstrumentationScopeName = "github.com/regularkevvv/agentic/otel"
	// SemanticConventionsRevision is the upstream standalone GenAI semantic
	// conventions commit this module implements. The upstream repository does
	// not publish a stable schema URL, so the module intentionally sets none.
	SemanticConventionsRevision = "a685613a207a580163353b8e48a7ad88967e7b42"
)

var (
	clientDurationBuckets = []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}
	tokenBuckets          = []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864}
	agentDurationBuckets  = []float64{0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6}
	callCountBuckets      = []float64{1, 2, 4, 8, 16, 32, 64, 128}
)

// Instrumentation implements agentic.Instrumentation with OTel traces,
// metrics, and event log records.
type Instrumentation struct {
	config config
	tracer trace.Tracer
	logger log.Logger

	clientTokenUsage metric.Int64Histogram
	clientDuration   metric.Float64Histogram
	clientTTFC       metric.Float64Histogram
	clientChunkTime  metric.Float64Histogram
	agentDuration    metric.Float64Histogram
	agentModelCalls  metric.Int64Histogram
	agentToolCalls   metric.Int64Histogram
	toolDuration     metric.Float64Histogram
}

// New constructs an optional Agentic observer. Applications remain
// responsible for configuring and shutting down their OTel SDK/exporters.
func New(options ...Option) (*Instrumentation, error) {
	configuration := config{
		tracerProvider: otelapi.GetTracerProvider(),
		meterProvider:  otelapi.GetMeterProvider(),
		loggerProvider: logglobal.GetLoggerProvider(),
	}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("agenticotel: option must not be nil")
		}
		if err := option(&configuration); err != nil {
			return nil, err
		}
	}
	version := instrumentationVersion()
	instrumentation := &Instrumentation{
		config: configuration,
		tracer: configuration.tracerProvider.Tracer(
			InstrumentationScopeName,
			trace.WithInstrumentationVersion(version),
		),
		logger: configuration.loggerProvider.Logger(
			InstrumentationScopeName,
			log.WithInstrumentationVersion(version),
		),
	}
	meter := configuration.meterProvider.Meter(
		InstrumentationScopeName,
		metric.WithInstrumentationVersion(version),
	)
	var errs []error
	instrumentation.clientTokenUsage, errs = intHistogram(meter, errs, "gen_ai.client.token.usage", "{token}", tokenBuckets)
	instrumentation.clientDuration, errs = floatHistogram(meter, errs, "gen_ai.client.operation.duration", "s", clientDurationBuckets)
	instrumentation.clientTTFC, errs = floatHistogram(meter, errs, "gen_ai.client.operation.time_to_first_chunk", "s", clientDurationBuckets)
	instrumentation.clientChunkTime, errs = floatHistogram(meter, errs, "gen_ai.client.operation.time_per_output_chunk", "s", clientDurationBuckets)
	instrumentation.agentDuration, errs = floatHistogram(meter, errs, "gen_ai.invoke_agent.duration", "s", agentDurationBuckets)
	instrumentation.agentModelCalls, errs = intHistogram(meter, errs, "gen_ai.invoke_agent.inference_calls", "{inference_call}", callCountBuckets)
	instrumentation.agentToolCalls, errs = intHistogram(meter, errs, "gen_ai.invoke_agent.tool_calls", "{tool_call}", callCountBuckets)
	instrumentation.toolDuration, errs = floatHistogram(meter, errs, "gen_ai.execute_tool.duration", "s", clientDurationBuckets)
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return instrumentation, nil
}

// MustNew is New with constructor errors converted to a panic.
func MustNew(options ...Option) *Instrumentation {
	instrumentation, err := New(options...)
	if err != nil {
		panic(err)
	}
	return instrumentation
}

func intHistogram(meter metric.Meter, errs []error, name, unit string, buckets []float64) (metric.Int64Histogram, []error) {
	histogram, err := meter.Int64Histogram(name, metric.WithUnit(unit), metric.WithExplicitBucketBoundaries(buckets...))
	if err != nil {
		errs = append(errs, err)
	}
	return histogram, errs
}

func floatHistogram(meter metric.Meter, errs []error, name, unit string, buckets []float64) (metric.Float64Histogram, []error) {
	histogram, err := meter.Float64Histogram(name, metric.WithUnit(unit), metric.WithExplicitBucketBoundaries(buckets...))
	if err != nil {
		errs = append(errs, err)
	}
	return histogram, errs
}

func instrumentationVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	if info.Main.Path == InstrumentationScopeName {
		return info.Main.Version
	}
	for _, dependency := range info.Deps {
		if dependency.Path == InstrumentationScopeName {
			return dependency.Version
		}
	}
	return ""
}

type invocationCounters struct {
	inference atomic.Int64
	tools     atomic.Int64
}

type invocationKey struct{}

type agentSpan struct {
	instrumentation *Instrumentation
	ctx             context.Context
	span            trace.Span
	start           time.Time
	operation       agentic.AgentOperation
	inputCount      int
	counters        *invocationCounters
}

func (i *Instrumentation) StartAgent(ctx context.Context, operation agentic.AgentOperation) (context.Context, agentic.AgentOperationSpan) {
	start := time.Now()
	name := "invoke_agent"
	if operation.Agent.Name != "" {
		name += " " + operation.Agent.Name
	}
	attrs := []attribute.KeyValue{attribute.String(keyOperation, "invoke_agent")}
	attrs = append(attrs, invokedAgentAttrs(operation.Agent)...)
	attrs = append(attrs, runAttrs(operation.Run)...)
	stringAttr(keyRequestModel, operation.ModelName, &attrs)
	attrs = append(attrs, agentRequestAttrs(operation.Request)...)
	stringAttr(keyExecutionMode, string(operation.Mode), &attrs)
	if i.config.messageContent {
		system, messages := i.messageContent(operation.Input)
		if value, ok := i.contentAttribute(keySystem, system); ok && len(system) > 0 {
			attrs = append(attrs, value)
		}
		if value, ok := i.contentAttribute(keyInputMessages, messages); ok && len(messages) > 0 {
			attrs = append(attrs, value)
		}
	}
	if i.config.toolContent && len(operation.Request.Tools) > 0 {
		if value, ok := i.contentAttribute(keyToolDefinitions, i.toolDefinitions(operation.Request.Tools)); ok {
			attrs = append(attrs, value)
		}
	}
	ctx, span := i.tracer.Start(ctx, name,
		trace.WithTimestamp(start),
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attrs...),
	)
	counters := &invocationCounters{}
	ctx = context.WithValue(ctx, invocationKey{}, counters)
	return ctx, &agentSpan{
		instrumentation: i, ctx: ctx, span: span, start: start, operation: operation,
		inputCount: len(operation.Input), counters: counters,
	}
}

func (s *agentSpan) End(result agentic.AgentOperationResult) {
	end := time.Now()
	defer s.span.End(trace.WithTimestamp(end))
	attrs := []attribute.KeyValue{attribute.String(keyExecutionOutcome, executionOutcome(result.Status))}
	attrs = append(attrs, usageAttrs(result.Usage)...)
	if result.FinishReason != "" {
		attrs = append(attrs, attribute.StringSlice(keyFinishReasons, []string{string(result.FinishReason)}))
	}
	if s.instrumentation.config.messageContent {
		messages := result.Messages
		if s.inputCount < len(messages) {
			messages = messages[s.inputCount:]
		} else {
			messages = nil
		}
		output := s.instrumentation.outputMessageContent(messages, result.FinishReason)
		if value, ok := s.instrumentation.contentAttribute(keyOutputMessages, output); ok && len(output) > 0 {
			attrs = append(attrs, value)
		}
	}
	if result.Error != nil {
		markSpanError(s.span, result.Error, s.instrumentation.config.exceptionContent)
	} else if result.Status == agentic.ExecutionFailed || result.Status == agentic.ExecutionInterrupted {
		kind := executionOutcome(result.Status)
		attrs = append(attrs, attribute.String(keyErrorType, kind))
		s.span.SetStatus(codes.Error, kind)
	}
	s.span.SetAttributes(attrs...)
	durationAttrs := agentNameMetricAttrs(s.operation.Agent)
	stringAttr(keyRequestModel, s.operation.ModelName, &durationAttrs)
	if result.Error != nil {
		durationAttrs = append(durationAttrs, attribute.String(keyErrorType, errorType(result.Error)))
	} else if result.Status == agentic.ExecutionFailed || result.Status == agentic.ExecutionInterrupted {
		durationAttrs = append(durationAttrs, attribute.String(keyErrorType, executionOutcome(result.Status)))
	}
	duration := end.Sub(s.start).Seconds()
	s.instrumentation.agentDuration.Record(s.ctx, duration, metric.WithAttributes(durationAttrs...))
	countAttrs := agentNameMetricAttrs(s.operation.Agent)
	s.instrumentation.agentModelCalls.Record(s.ctx, s.counters.inference.Load(), metric.WithAttributes(countAttrs...))
	s.instrumentation.agentToolCalls.Record(s.ctx, s.counters.tools.Load(), metric.WithAttributes(countAttrs...))
}

type modelSpan struct {
	instrumentation *Instrumentation
	ctx             context.Context
	span            trace.Span
	start           time.Time
	operation       agentic.ModelOperation

	mu         sync.Mutex
	firstChunk time.Time
	lastChunk  time.Time
}

func (i *Instrumentation) StartModelRequest(ctx context.Context, operation agentic.ModelOperation) (context.Context, agentic.ModelOperationSpan) {
	start := time.Now()
	if counters, ok := ctx.Value(invocationKey{}).(*invocationCounters); ok {
		counters.inference.Add(1)
	}
	attrs := modelAttrs(operation.Model, operation.Request.Model)
	attrs = append(attrs, requestAttrs(operation.Request)...)
	attrs = append(attrs, runAttrs(operation.Run)...)
	attrs = append(attrs, agentContextAttrs(operation.Agent)...)
	if i.config.messageContent {
		system, messages := i.messageContent(operation.Request.Messages)
		if value, ok := i.contentAttribute(keySystem, system); ok && len(system) > 0 {
			attrs = append(attrs, value)
		}
		if value, ok := i.contentAttribute(keyInputMessages, messages); ok && len(messages) > 0 {
			attrs = append(attrs, value)
		}
	}
	if i.config.toolContent && len(operation.Request.Tools) > 0 {
		definitions := i.toolDefinitions(operation.Request.Tools)
		if value, ok := i.contentAttribute(keyToolDefinitions, definitions); ok {
			attrs = append(attrs, value)
		}
	}
	kind := trace.SpanKindClient
	if operation.Model.InProcess {
		kind = trace.SpanKindInternal
	}
	operationName := operation.Model.Operation
	if operationName == "" {
		operationName = "chat"
	}
	name := operationName + " " + operation.Request.Model
	ctx, span := i.tracer.Start(ctx, name,
		trace.WithTimestamp(start),
		trace.WithSpanKind(kind),
		trace.WithAttributes(attrs...),
	)
	return ctx, &modelSpan{instrumentation: i, ctx: ctx, span: span, start: start, operation: operation}
}

func (s *modelSpan) ObserveStreamEvent(event agentic.StreamEvent) {
	if !isOutputChunk(event.Type) {
		return
	}
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	attrs := modelMetricAttrs(s.operation, nil, "")
	if s.firstChunk.IsZero() {
		s.firstChunk = now
		s.lastChunk = now
		s.instrumentation.clientTTFC.Record(s.ctx, now.Sub(s.start).Seconds(), metric.WithAttributes(attrs...))
		return
	}
	s.instrumentation.clientChunkTime.Record(s.ctx, now.Sub(s.lastChunk).Seconds(), metric.WithAttributes(attrs...))
	s.lastChunk = now
}

func isOutputChunk(eventType agentic.StreamEventType) bool {
	switch eventType {
	case agentic.StreamEventTextDelta, agentic.StreamEventThinkingDelta,
		agentic.StreamEventToolCallStart, agentic.StreamEventToolCallDelta:
		return true
	default:
		return false
	}
}

func (s *modelSpan) End(result agentic.ModelOperationResult) {
	end := time.Now()
	defer s.span.End(trace.WithTimestamp(end))
	attrs := responseAttrs(result.Response)
	s.mu.Lock()
	if !s.firstChunk.IsZero() {
		attrs = append(attrs, attribute.Float64(keyTTFC, s.firstChunk.Sub(s.start).Seconds()))
	}
	s.mu.Unlock()
	if s.instrumentation.config.messageContent && result.Response != nil {
		messages := s.instrumentation.outputMessageContent([]agentic.Message{result.Response.Message}, result.Response.FinishReason)
		if value, ok := s.instrumentation.contentAttribute(keyOutputMessages, messages); ok && len(messages) > 0 {
			attrs = append(attrs, value)
		}
	}
	errorKind := ""
	if result.Error != nil {
		errorKind = markSpanError(s.span, result.Error, s.instrumentation.config.exceptionContent)
		s.instrumentation.emitException(s.ctx, result.Error)
	} else if result.Response != nil && result.Response.FinishReason == agentic.FinishReasonError {
		errorKind = "provider_error"
		s.span.SetAttributes(attribute.String(keyErrorType, errorKind))
		s.span.SetStatus(codes.Error, errorKind)
		s.instrumentation.emitExceptionType(s.ctx, errorKind)
	}
	s.span.SetAttributes(attrs...)
	durationAttrs := modelMetricAttrs(s.operation, result.Response, errorKind)
	s.instrumentation.clientDuration.Record(s.ctx, end.Sub(s.start).Seconds(), metric.WithAttributes(durationAttrs...))
	if result.Response != nil {
		usageMetricAttrs := modelMetricAttrs(s.operation, result.Response, "")
		if result.Response.Usage.PromptTokens > 0 {
			tokenAttrs := append(append([]attribute.KeyValue(nil), usageMetricAttrs...), attribute.String("gen_ai.token.type", "input"))
			s.instrumentation.clientTokenUsage.Record(s.ctx, int64(result.Response.Usage.PromptTokens), metric.WithAttributes(tokenAttrs...))
		}
		if result.Response.Usage.CompletionTokens > 0 {
			tokenAttrs := append(append([]attribute.KeyValue(nil), usageMetricAttrs...), attribute.String("gen_ai.token.type", "output"))
			s.instrumentation.clientTokenUsage.Record(s.ctx, int64(result.Response.Usage.CompletionTokens), metric.WithAttributes(tokenAttrs...))
		}
	}
	if s.instrumentation.config.inferenceDetails {
		s.instrumentation.emitInferenceDetails(s.ctx, s.operation, result)
	}
}

type toolSpan struct {
	instrumentation *Instrumentation
	ctx             context.Context
	span            trace.Span
	start           time.Time
	operation       agentic.ToolOperation
}

func (i *Instrumentation) StartTool(ctx context.Context, operation agentic.ToolOperation) (context.Context, agentic.ToolOperationSpan) {
	start := time.Now()
	if counters, ok := ctx.Value(invocationKey{}).(*invocationCounters); ok {
		counters.tools.Add(1)
	}
	attrs := []attribute.KeyValue{
		attribute.String(keyOperation, "execute_tool"),
		attribute.String(keyToolName, operation.Call.Name),
		attribute.String(keyToolType, "function"),
	}
	stringAttr(keyToolCallID, operation.Call.ID, &attrs)
	stringAttr(keyAgentName, operation.Agent.Name, &attrs)
	attrs = append(attrs, runAttrs(operation.Run)...)
	if operation.Attempt > 0 {
		attrs = append(attrs, attribute.Int(keyToolAttempt, operation.Attempt))
	}
	if operation.HandlerResumed {
		attrs = append(attrs, attribute.Bool(keyToolResumed, true))
	}
	if i.config.toolContent {
		if description, ok := i.filtered(ContentToolDescription, operation.Definition.Function.Description); ok && description != "" {
			attrs = append(attrs, attribute.String("gen_ai.tool.description", description))
		}
		if arguments, ok := i.filteredJSONObject(ContentToolArguments, operation.Call.Input); ok {
			if value, ok := i.contentAttribute(keyToolArguments, arguments); ok {
				attrs = append(attrs, value)
			}
		}
	}
	ctx, span := i.tracer.Start(ctx, "execute_tool "+operation.Call.Name,
		trace.WithTimestamp(start),
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attrs...),
	)
	return ctx, &toolSpan{instrumentation: i, ctx: ctx, span: span, start: start, operation: operation}
}

func (s *toolSpan) End(result agentic.ToolOperationResult) {
	end := time.Now()
	defer s.span.End(trace.WithTimestamp(end))
	errorKind := ""
	toolErr := result.Error
	if toolErr == nil {
		toolErr = result.Result.Error
	}
	if toolErr != nil {
		errorKind = markSpanError(s.span, toolErr, s.instrumentation.config.exceptionContent)
	} else if result.Result.IsError {
		errorKind = "tool_error"
		s.span.SetAttributes(attribute.String(keyErrorType, errorKind))
		s.span.SetStatus(codes.Error, errorKind)
	}
	if result.Suspended {
		s.span.SetAttributes(attribute.String(keyToolOutcome, "suspended"))
	}
	// gen_ai.tool.call.result describes a successful execution result. Error
	// payloads remain available to Agentic, but are not mislabeled as a result
	// by the semantic-convention projection.
	if s.instrumentation.config.toolContent && errorKind == "" && !result.Suspended {
		if filtered, ok := s.instrumentation.filteredJSONObject(ContentToolResult, result.Result.Content); ok {
			if value, ok := s.instrumentation.contentAttribute(keyToolResult, filtered); ok {
				s.span.SetAttributes(value)
			}
		}
	}
	attrs := toolMetricAttrs(s.operation)
	if errorKind != "" {
		attrs = append(attrs, attribute.String(keyErrorType, errorKind))
	}
	s.instrumentation.toolDuration.Record(s.ctx, end.Sub(s.start).Seconds(), metric.WithAttributes(attrs...))
}

func usageAttrs(usage agentic.Usage) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 4)
	if usage.PromptTokens > 0 {
		attrs = append(attrs, attribute.Int(keyInputTokens, usage.PromptTokens))
	}
	if usage.CompletionTokens > 0 {
		attrs = append(attrs, attribute.Int(keyOutputTokens, usage.CompletionTokens))
	}
	if usage.CacheReadTokens > 0 {
		attrs = append(attrs, attribute.Int(keyCacheRead, usage.CacheReadTokens))
	}
	if usage.CacheCreationTokens > 0 {
		attrs = append(attrs, attribute.Int(keyCacheCreation, usage.CacheCreationTokens))
	}
	return attrs
}

func agentNameMetricAttrs(identity agentic.AgentIdentity) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 1)
	stringAttr(keyAgentName, identity.Name, &attrs)
	return attrs
}

func modelMetricAttrs(operation agentic.ModelOperation, response *agentic.ChatResponse, errorKind string) []attribute.KeyValue {
	attrs := modelAttrs(operation.Model, operation.Request.Model)
	if response != nil {
		stringAttr(keyResponseModel, response.Model, &attrs)
	}
	if errorKind != "" {
		attrs = append(attrs, attribute.String(keyErrorType, errorKind))
	}
	return attrs
}

func toolMetricAttrs(operation agentic.ToolOperation) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.String(keyToolName, operation.Call.Name),
		attribute.String(keyToolType, "function"),
	}
	stringAttr(keyAgentName, operation.Agent.Name, &attrs)
	return attrs
}

func executionOutcome(status agentic.ExecutionStatus) string {
	switch status {
	case agentic.ExecutionCompleted:
		return "completed"
	case agentic.ExecutionSuspended:
		return "suspended"
	case agentic.ExecutionStopped:
		return "stopped"
	case agentic.ExecutionInterrupted:
		return "interrupted"
	case agentic.ExecutionFailed:
		return "failed"
	default:
		return "unknown"
	}
}

var _ agentic.Instrumentation = (*Instrumentation)(nil)
