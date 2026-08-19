package agentic

import (
	"context"
	"encoding/json"
)

// AgentIdentity is the application-owned identity of an agent definition.
type AgentIdentity struct {
	Name        string
	Description string
	Version     string
}

// RunMetadata contains correlation identifiers owned by a higher-level
// runtime. ConversationID maps to gen_ai.conversation.id; RunID is an
// Agentic-specific durable execution identifier.
type RunMetadata struct {
	ConversationID string
	RunID          string
}

// AgentInvocationMode identifies the public execution entry point.
type AgentInvocationMode string

const (
	AgentInvocationStart    AgentInvocationMode = "start"
	AgentInvocationContinue AgentInvocationMode = "continue"
	AgentInvocationResume   AgentInvocationMode = "resume"
)

// AgentOperation describes one Drive, Resume, or RunStream invocation.
type AgentOperation struct {
	Agent     AgentIdentity
	Model     ModelMetadata
	ModelName string
	// Request contains the stable request settings and tool definitions known
	// at invocation start. Messages are carried separately in Input. Tools are
	// omitted when a dynamic tool-preparation callback can change them per turn.
	Request ChatRequest
	Run     RunMetadata
	Mode    AgentInvocationMode
	Input   []Message
}

// AgentOperationResult is the terminal state of one invocation. Suspensions
// close the invocation normally and can be resumed by a later invocation.
type AgentOperationResult struct {
	Status       ExecutionStatus
	Messages     []Message
	Usage        Usage
	FinishReason FinishReason
	Error        error
}

// ModelOperation describes one logical provider request.
type ModelOperation struct {
	Agent AgentIdentity
	Model ModelMetadata
	Run   RunMetadata
	// Request is a defensive observation copy. Opaque ProviderOptions are
	// deliberately omitted because they may contain secrets, non-JSON values,
	// or mutable provider-owned state.
	Request   ChatRequest
	Iteration int
}

// ModelOperationResult is the terminal provider response or error.
type ModelOperationResult struct {
	Response *ChatResponse
	Error    error
}

// ToolOperation describes one admitted handler invocation.
type ToolOperation struct {
	Agent      AgentIdentity
	Run        RunMetadata
	Call       ToolUse
	Definition Tool
	Attempt    int
	// HandlerResumed is true only when a suspendable handler is re-entered with
	// ToolResumeContext. A call admitted after an approval-gate suspension is a
	// fresh handler execution and remains false.
	HandlerResumed bool
}

// ToolOperationResult is the raw handler result before result processors run.
type ToolOperationResult struct {
	Result    ToolExecutionResult
	Error     error
	Suspended bool
}

// Instrumentation is the dependency-neutral lifecycle observed by optional
// telemetry adapters. Implementations must treat values as immutable. Agentic
// isolates callback panics and ignores nil returned contexts and spans.
type Instrumentation interface {
	StartAgent(context.Context, AgentOperation) (context.Context, AgentOperationSpan)
	StartModelRequest(context.Context, ModelOperation) (context.Context, ModelOperationSpan)
	StartTool(context.Context, ToolOperation) (context.Context, ToolOperationSpan)
}

type AgentOperationSpan interface {
	End(AgentOperationResult)
}

type ModelOperationSpan interface {
	ObserveStreamEvent(StreamEvent)
	End(ModelOperationResult)
}

type ToolOperationSpan interface {
	End(ToolOperationResult)
}

type noopAgentOperationSpan struct{}
type noopModelOperationSpan struct{}
type noopToolOperationSpan struct{}

func (noopAgentOperationSpan) End(AgentOperationResult)       {}
func (noopModelOperationSpan) ObserveStreamEvent(StreamEvent) {}
func (noopModelOperationSpan) End(ModelOperationResult)       {}
func (noopToolOperationSpan) End(ToolOperationResult)         {}

type inheritedInstrumentation struct {
	observer Instrumentation
	run      RunMetadata
}

type inheritedInstrumentationKey struct{}

func instrumentationFromContext(ctx context.Context) inheritedInstrumentation {
	value, _ := ctx.Value(inheritedInstrumentationKey{}).(inheritedInstrumentation)
	return value
}

func withInheritedInstrumentation(ctx context.Context, observer Instrumentation, run RunMetadata) context.Context {
	return context.WithValue(ctx, inheritedInstrumentationKey{}, inheritedInstrumentation{observer: observer, run: run})
}

func safeStartAgent(ctx context.Context, observer Instrumentation, operation AgentOperation) (out context.Context, span AgentOperationSpan) {
	out, span = ctx, noopAgentOperationSpan{}
	if observer == nil {
		return out, span
	}
	defer func() {
		if recover() != nil {
			out, span = ctx, noopAgentOperationSpan{}
		}
		if out == nil {
			out = ctx
		}
		if span == nil {
			span = noopAgentOperationSpan{}
		}
	}()
	out, span = observer.StartAgent(ctx, cloneAgentOperation(operation))
	return out, span
}

func safeStartModel(ctx context.Context, observer Instrumentation, operation ModelOperation) (out context.Context, span ModelOperationSpan) {
	out, span = ctx, noopModelOperationSpan{}
	if observer == nil {
		return out, span
	}
	defer func() {
		if recover() != nil {
			out, span = ctx, noopModelOperationSpan{}
		}
		if out == nil {
			out = ctx
		}
		if span == nil {
			span = noopModelOperationSpan{}
		}
	}()
	out, span = observer.StartModelRequest(ctx, cloneModelOperation(operation))
	return out, span
}

func safeStartTool(ctx context.Context, observer Instrumentation, operation ToolOperation) (out context.Context, span ToolOperationSpan) {
	out, span = ctx, noopToolOperationSpan{}
	if observer == nil {
		return out, span
	}
	defer func() {
		if recover() != nil {
			out, span = ctx, noopToolOperationSpan{}
		}
		if out == nil {
			out = ctx
		}
		if span == nil {
			span = noopToolOperationSpan{}
		}
	}()
	out, span = observer.StartTool(ctx, cloneToolOperation(operation))
	return out, span
}

func safeEndAgent(span AgentOperationSpan, result AgentOperationResult) {
	defer func() { _ = recover() }()
	span.End(cloneAgentOperationResult(result))
}

func safeObserveStreamEvent(span ModelOperationSpan, event StreamEvent) {
	defer func() { _ = recover() }()
	span.ObserveStreamEvent(cloneStreamEvent(event))
}

func safeEndModel(span ModelOperationSpan, result ModelOperationResult) {
	defer func() { _ = recover() }()
	span.End(cloneModelOperationResult(result))
}

func safeEndTool(span ToolOperationSpan, result ToolOperationResult) {
	defer func() { _ = recover() }()
	span.End(cloneToolOperationResult(result))
}

func cloneAgentOperation(operation AgentOperation) AgentOperation {
	operation.Input = cloneInstrumentationMessages(operation.Input)
	operation.Request = cloneChatRequest(operation.Request)
	return operation
}

func cloneAgentOperationResult(result AgentOperationResult) AgentOperationResult {
	result.Messages = cloneInstrumentationMessages(result.Messages)
	result.Usage.RequestUsages = append([]RequestUsage(nil), result.Usage.RequestUsages...)
	return result
}

func cloneModelOperation(operation ModelOperation) ModelOperation {
	operation.Request = cloneChatRequest(operation.Request)
	return operation
}

func cloneModelOperationResult(result ModelOperationResult) ModelOperationResult {
	if result.Response != nil {
		copy := *result.Response
		if messages := cloneInstrumentationMessages([]Message{result.Response.Message}); len(messages) > 0 {
			copy.Message = messages[0]
		} else {
			copy.Message = Message{}
		}
		copy.Usage.RequestUsages = append([]RequestUsage(nil), result.Response.Usage.RequestUsages...)
		result.Response = &copy
	}
	return result
}

func cloneToolOperation(operation ToolOperation) ToolOperation {
	operation.Call.Input = cloneJSONValue(operation.Call.Input)
	operation.Definition = cloneJSONValue(operation.Definition)
	return operation
}

func cloneToolOperationResult(result ToolOperationResult) ToolOperationResult {
	result.Result.Content = cloneJSONValue(result.Result.Content)
	return result
}

func cloneChatRequest(request ChatRequest) ChatRequest {
	request.Messages = cloneInstrumentationMessages(request.Messages)
	request.Tools = cloneJSONValue(request.Tools)
	request.StopSequences = append([]string(nil), request.StopSequences...)
	request.Temperature = clonePointer(request.Temperature)
	request.MaxTokens = clonePointer(request.MaxTokens)
	request.TopP = clonePointer(request.TopP)
	request.ToolChoice = clonePointer(request.ToolChoice)
	request.ResponseFormat = cloneJSONValue(request.ResponseFormat)
	request.Thinking = clonePointer(request.Thinking)
	request.PromptCache = clonePromptCache(request.PromptCache)
	request.ProviderOptions = nil // opaque provider values are not observation data
	return request
}

func clonePointer[T any](value *T) *T {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func cloneStreamEvent(event StreamEvent) StreamEvent {
	if event.ToolUse != nil {
		copy := *event.ToolUse
		copy.Input = cloneJSONValue(event.ToolUse.Input)
		event.ToolUse = &copy
	}
	if event.Usage != nil {
		copy := *event.Usage
		copy.RequestUsages = append([]RequestUsage(nil), event.Usage.RequestUsages...)
		event.Usage = &copy
	}
	return event
}

// Observation payloads are JSON-shaped by contract. If application-owned
// provider metadata makes a message non-serializable, omit it from the
// callback rather than exposing mutable execution state to an observer.
func cloneInstrumentationMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return nil
	}
	return cloneJSONValue(messages)
}

func cloneJSONValue[T any](value T) T {
	encoded, err := json.Marshal(value)
	if err != nil {
		var zero T
		return zero
	}
	var copy T
	if json.Unmarshal(encoded, &copy) != nil {
		var zero T
		return zero
	}
	return copy
}
