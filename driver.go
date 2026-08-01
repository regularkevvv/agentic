package agentic

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
)

// Driver is the resumable execution capability implemented by built-in agents
// and bound runners. Runner remains the small, source-compatible convenience
// interface; callers that need session control opt into Driver explicitly.
type Driver[O any] interface {
	Runner[O]
	Drive(context.Context, DriveInput, ...RunOption) (*Execution[O], error)
	Resume(context.Context, ResumeInput, ...RunOption) (*Execution[O], error)
}

// DriveMode selects whether a driver starts a fresh user turn or continues a
// fully paired transcript without adding a synthetic prompt.
type DriveMode uint8

const (
	DriveStart DriveMode = iota
	DriveContinue
)

// DriveInput is the explicit input to a driver execution.
//
// DriveStart requires Prompt to be a user message. DriveContinue forbids a
// prompt and continues directly from History. History is copied before any
// model or tool work begins.
type DriveInput struct {
	Mode    DriveMode
	History []Message
	Prompt  *Message
}

// ResumeInput resolves a suspended executable tool frontier. Prompt, when
// supplied, is appended only after all resolved tool results.
type ResumeInput struct {
	History    []Message
	Suspension Suspension
	Decisions  []ToolResumeDecision
	Prompt     *Message
}

// ToolResumeDecision resolves one executable call from a suspension.
type ToolResumeDecision struct {
	CallID string
	Action ToolResumeAction
	// Input nil means use the original persisted tool arguments.
	Input  map[string]any
	Result *ToolExecutionResult
	// Payload is opaque handler-owned resume data. The driver never interprets
	// it; it is exposed only through CurrentToolResume while re-entering a
	// suspendable handler. It is valid only with ToolResumeExecute.
	Payload json.RawMessage
}

// ToolResumeAction describes how a suspended tool call is completed.
type ToolResumeAction uint8

const (
	ToolResumeInvalid ToolResumeAction = iota
	ToolResumeExecute
	ToolResumeReturn
)

// Execution contains either a completed result or the durable partial state
// reached before a failure, stop, interruption, or suspension.
type Execution[O any] struct {
	Status     ExecutionStatus
	Result     *Result[O]
	Suspension *Suspension
}

// RequireDriver returns the execution driver capability for a ready-to-run
// runner. It never reconstructs an agent loop from Runner.Run.
func RequireDriver[O any](runner Runner[O]) (Driver[O], error) {
	driver, ok := runner.(Driver[O])
	if !ok || isNilDriver(driver) {
		return nil, ErrDriverRequired
	}
	return driver, nil
}

func isNilDriver[O any](driver Driver[O]) bool {
	value := reflect.ValueOf(driver)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func executionError[O any](execution *Execution[O], err error) error {
	if err != nil {
		return err
	}
	if execution == nil {
		return nil
	}
	switch execution.Status {
	case ExecutionSuspended:
		return ErrExecutionSuspended
	case ExecutionStopped:
		return ErrExecutionStopped
	case ExecutionInterrupted:
		return ErrExecutionInterrupted
	case ExecutionFailed:
		return ErrExecutionFailed
	default:
		return nil
	}
}

func newSuspensionID() string {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err == nil {
		return hex.EncodeToString(bytes[:])
	}
	// Failure to obtain random bytes is exceptionally unusual. The suspension
	// ID is a stale-state guard rather than an authentication credential, so a
	// deterministic fallback is still safer than failing after a gate has
	// already admitted a suspension.
	return fmt.Sprintf("suspension-%p", &bytes)
}

// ToolCallContext identifies the handler invocation currently associated with
// a context. It is available to plain and dependency-aware tools alike.
type ToolCallContext struct {
	ID      string
	Name    string
	Attempt int
}

// ToolResumeContext is attached only while Driver.Resume re-enters a handler
// that previously returned ToolHandlerSuspension. Deferral is the exact
// handler-owned value carried by the durable root suspension. Payload is the
// caller-validated opaque data from ToolResumeDecision.
type ToolResumeContext struct {
	SuspensionID string
	Deferral     ToolDeferral
	Payload      json.RawMessage
}

type toolCallContextKey struct{}
type toolResumeContextKey struct{}

// WithToolCallContext starts a fresh logical tool invocation. It installs call
// metadata for CurrentToolCall and clears any handler-owned resume metadata
// inherited from an outer composite tool. Driver-managed handlers receive this
// context automatically; composite tool hosts use it before invoking a nested
// handler.
func WithToolCallContext(ctx context.Context, call ToolCallContext) context.Context {
	ctx = context.WithValue(ctx, toolResumeContextKey{}, struct{}{})
	return context.WithValue(ctx, toolCallContextKey{}, call)
}

// CurrentToolCall returns public metadata for the tool invocation associated
// with ctx.
func CurrentToolCall(ctx context.Context) (ToolCallContext, bool) {
	call, ok := ctx.Value(toolCallContextKey{}).(ToolCallContext)
	return call, ok
}

func withToolResumeContext(ctx context.Context, resume ToolResumeContext) context.Context {
	resume.Deferral.Payload = append(json.RawMessage(nil), resume.Deferral.Payload...)
	resume.Payload = append(json.RawMessage(nil), resume.Payload...)
	return context.WithValue(ctx, toolResumeContextKey{}, resume)
}

// CurrentToolResume returns the exact resume metadata for the current handler.
// Ordinary handler execution returns false.
func CurrentToolResume(ctx context.Context) (ToolResumeContext, bool) {
	resume, ok := ctx.Value(toolResumeContextKey{}).(ToolResumeContext)
	if !ok {
		return ToolResumeContext{}, false
	}
	resume.Deferral.Payload = append(json.RawMessage(nil), resume.Deferral.Payload...)
	resume.Payload = append(json.RawMessage(nil), resume.Payload...)
	return resume, true
}

func makeToolResultError(call ToolUse, message string, cause error) ToolExecutionResult {
	if cause == nil {
		cause = errors.New(message)
	}
	return ToolExecutionResult{
		ToolUseID: call.ID,
		ToolName:  call.Name,
		Content:   message,
		IsError:   true,
		Error:     cause,
	}
}
