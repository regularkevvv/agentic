package agentic

import (
	"context"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
	agentictool "github.com/regularkevvv/agentic/tool"
)

// ---------------------------------------------------------------------------
// Domain type aliases (from core)
// These allow users to import only the root "agentic" package for all types.
// ---------------------------------------------------------------------------

type ToolType = core.ToolType
type Tool = core.Tool
type Function = core.Function
type ToolChoice = core.ToolChoice
type ChatRequest = core.ChatRequest
type ChatResponse = core.ChatResponse
type Choice = core.Choice
type FinishReason = core.FinishReason
type Usage = core.Usage
type RequestUsage = core.RequestUsage
type MessageRole = core.MessageRole
type ContentType = core.ContentType
type Message = core.Message
type Part = core.Part
type ImageURL = core.ImageURL
type ToolUse = core.ToolUse
type ToolResult = core.ToolResult
type Model = core.Model
type StreamModel = core.StreamModel
type RunContext[D any] = core.RunContext[D]
type dependencyEnvelope = core.DependencyEnvelope
type ToolHandler = core.ToolHandler
type ToolExecutionResult = core.ToolExecutionResult
type ToolConfig = core.ToolConfig
type ResponseFormat = core.ResponseFormat
type JSONSchemaFormat = core.JSONSchemaFormat
type ThinkingConfig = core.ThinkingConfig
type ThinkingBlock = core.ThinkingBlock
type ImageData = core.ImageData
type AudioURL = core.AudioURL
type VideoURL = core.VideoURL
type DocumentURL = core.DocumentURL
type CachePoint = core.CachePoint
type UploadedFile = core.UploadedFile
type StreamResult = core.StreamResult
type StreamEvent = core.StreamEvent
type StreamEventType = core.StreamEventType
type OutputMode = core.OutputMode

// Domain constants (from core)
const (
	ToolTypeFunction          = core.ToolTypeFunction
	ToolChoiceNone            = core.ToolChoiceNone
	ToolChoiceAuto            = core.ToolChoiceAuto
	ToolChoiceRequired        = core.ToolChoiceRequired
	FinishReasonStop          = core.FinishReasonStop
	FinishReasonLength        = core.FinishReasonLength
	FinishReasonToolCalls     = core.FinishReasonToolCalls
	FinishReasonContentFilter = core.FinishReasonContentFilter
	RoleSystem                = core.RoleSystem
	RoleUser                  = core.RoleUser
	RoleAssistant             = core.RoleAssistant
	RoleTool                  = core.RoleTool
	ContentText               = core.ContentText
	ContentImageURL           = core.ContentImageURL
	ContentToolUse            = core.ContentToolUse
	ContentToolResult         = core.ContentToolResult
	ContentThinking           = core.ContentThinking
	ContentImageData          = core.ContentImageData
	ContentAudioURL           = core.ContentAudioURL
	ContentVideoURL           = core.ContentVideoURL
	ContentDocumentURL        = core.ContentDocumentURL
	ContentCachePoint         = core.ContentCachePoint
	ContentUploadedFile       = core.ContentUploadedFile
	OutputModeTool            = core.OutputModeTool
	OutputModeNative          = core.OutputModeNative
	OutputModePrompted        = core.OutputModePrompted
	OutputModeText            = core.OutputModeText
)

// Stream event constants
const (
	StreamEventTextDelta     = core.StreamEventTextDelta
	StreamEventToolCallStart = core.StreamEventToolCallStart
	StreamEventToolCallDelta = core.StreamEventToolCallDelta
	StreamEventToolResult    = core.StreamEventToolResult
	StreamEventDone          = core.StreamEventDone
	StreamEventError         = core.StreamEventError
	StreamEventThinkingDelta = core.StreamEventThinkingDelta
)

// ---------------------------------------------------------------------------
// Tool type aliases (from tool/)
// ---------------------------------------------------------------------------

type ToolRegistry = agentictool.ToolRegistry
type Toolset = agentictool.Toolset
type FuncToolset = agentictool.FuncToolset
type PlainToolHandler[TInput any, TOutput any] = agentictool.PlainToolHandler[TInput, TOutput]
type ContextToolHandler[TInput any, TOutput any] = agentictool.ContextToolHandler[TInput, TOutput]
type DepsToolHandler[TInput any, TOutput any, DepsT any] = agentictool.DepsToolHandler[TInput, TOutput, DepsT]
type ToolOption = agentictool.ToolOption
type AutoToolOption = agentictool.AutoToolOption

// Channel-backed and approval tool types (from tool/).
type ApprovalFunc = agentictool.ApprovalFunc
type ChannelToolOption = agentictool.ChannelToolOption

// ---------------------------------------------------------------------------
// Convenience functions — core domain
// ---------------------------------------------------------------------------

func NewTextMessage(role MessageRole, text string) Message {
	return core.NewTextMessage(role, text)
}

func NewToolUseMessage(toolUses ...ToolUse) Message {
	return core.NewToolUseMessage(toolUses...)
}

func NewToolResultMessage(toolUseID string, content string, isError bool) Message {
	return core.NewToolResultMessage(toolUseID, content, isError)
}

func FormatToolResult(result interface{}) string {
	return core.FormatToolResult(result)
}

func NewStreamResult(ch <-chan StreamEvent) *StreamResult {
	return core.NewStreamResult(ch)
}

// ---------------------------------------------------------------------------
// Convenience functions — tool building
// ---------------------------------------------------------------------------

func NewToolFromStruct(name, description string, input interface{}) (Tool, error) {
	return agentictool.NewToolFromStruct(name, description, input)
}

func MustNewToolFromStruct(name, description string, input interface{}) Tool {
	return agentictool.MustNewToolFromStruct(name, description, input)
}

func ToolPlain[TInput any, TOutput any](
	name, description string,
	handler func(input TInput) (TOutput, error),
	opts ...ToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.ToolPlain(name, description, handler, opts...)
}

func MustToolPlain[TInput any, TOutput any](
	name, description string,
	handler func(input TInput) (TOutput, error),
	opts ...ToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustToolPlain(name, description, handler, opts...)
}

func ToolWithContext[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...ToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.ToolWithContext(name, description, handler, opts...)
}

func MustToolWithContext[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...ToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustToolWithContext(name, description, handler, opts...)
}

func ToolWithDeps[TInput any, TOutput any, DepsT any](
	name, description string,
	handler func(ctx RunContext[DepsT], input TInput) (TOutput, error),
	opts ...ToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.ToolWithDeps[TInput, TOutput, DepsT](name, description, handler, opts...)
}

func MustToolWithDeps[TInput any, TOutput any, DepsT any](
	name, description string,
	handler func(ctx RunContext[DepsT], input TInput) (TOutput, error),
	opts ...ToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustToolWithDeps[TInput, TOutput, DepsT](name, description, handler, opts...)
}

func NewRegistry() ToolRegistry {
	return agentictool.NewRegistry()
}

func NewToolset() *FuncToolset {
	return agentictool.NewToolset()
}

func CombineToolsets(sets ...Toolset) Toolset {
	return agentictool.CombineToolsets(sets...)
}

func FilterToolset(set Toolset, predicate func(toolName string) bool) Toolset {
	return agentictool.FilterToolset(set, predicate)
}

func PrefixToolset(set Toolset, prefix string) Toolset {
	return agentictool.PrefixToolset(set, prefix)
}

func RegisterToolset(registry ToolRegistry, set Toolset) error {
	return agentictool.RegisterToolset(registry, set)
}

// WithToolMaxRetries sets the max retry count for a specific tool.
func WithToolMaxRetries(n int) ToolOption {
	return agentictool.WithToolMaxRetries(n)
}

// ---------------------------------------------------------------------------
// Auto-registration convenience functions
// ---------------------------------------------------------------------------

func AutoToolName(name string) AutoToolOption {
	return agentictool.WithName(name)
}

func AutoToolDescription(desc string) AutoToolOption {
	return agentictool.WithDescription(desc)
}

func AutoTool[TInput any, TOutput any](
	handler func(input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.Auto(handler, opts...)
}

func MustAutoTool[TInput any, TOutput any](
	handler func(input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustAuto(handler, opts...)
}

func AutoToolWithContext[TInput any, TOutput any](
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.AutoWithContext(handler, opts...)
}

func MustAutoToolWithContext[TInput any, TOutput any](
	handler func(ctx context.Context, input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustAutoWithContext(handler, opts...)
}

func AutoToolWithDeps[TInput any, TOutput any, DepsT any](
	handler func(ctx RunContext[DepsT], input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.AutoWithDeps[TInput, TOutput, DepsT](handler, opts...)
}

func MustAutoToolWithDeps[TInput any, TOutput any, DepsT any](
	handler func(ctx RunContext[DepsT], input TInput) (TOutput, error),
	opts ...AutoToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustAutoWithDeps[TInput, TOutput, DepsT](handler, opts...)
}

// ---------------------------------------------------------------------------
// Statically checked agent tool registration
// ---------------------------------------------------------------------------

type toolTarget interface {
	registerTool(Tool, ToolHandler)
}

type dependencyToolTarget[D any] interface {
	toolTarget
	dependencyType(D)
}

// AddToolPlain registers a context-free handler on any agent facade.
func AddToolPlain[T toolTarget, TInput, TOutput any](
	target T,
	handler func(TInput) (TOutput, error),
	opts ...AutoToolOption,
) T {
	tool, h := agentictool.MustAuto(handler, opts...)
	target.registerTool(tool, h)
	return target
}

// AddTool registers the standard context-aware handler on any agent facade.
func AddTool[T toolTarget, TInput, TOutput any](
	target T,
	handler func(context.Context, TInput) (TOutput, error),
	opts ...AutoToolOption,
) T {
	tool, h := agentictool.MustAutoWithContext(handler, opts...)
	target.registerTool(tool, h)
	return target
}

// AddToolWithContext is an explicit spelling of AddTool.
func AddToolWithContext[T toolTarget, TInput, TOutput any](
	target T,
	handler func(context.Context, TInput) (TOutput, error),
	opts ...AutoToolOption,
) T {
	return AddTool(target, handler, opts...)
}

// AddToolWithDeps registers a handler only when its D exactly matches the
// dependency-aware facade's D.
func AddToolWithDeps[D, TInput, TOutput any, T dependencyToolTarget[D]](
	target T,
	handler func(RunContext[D], TInput) (TOutput, error),
	opts ...AutoToolOption,
) T {
	tool, h := agentictool.MustAutoWithDeps[TInput, TOutput, D](handler, opts...)
	target.registerTool(tool, h)
	return target
}

// ---------------------------------------------------------------------------
// Channel-backed and approval tool convenience functions
// ---------------------------------------------------------------------------

func ChannelTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (<-chan TOutput, error),
	opts ...ChannelToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.ChannelTool(name, description, handler, opts...)
}

// MustChannelTool is like ChannelTool but panics on definition errors.
func MustChannelTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (<-chan TOutput, error),
	opts ...ChannelToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustChannelTool(name, description, handler, opts...)
}

// ApprovalTool creates a synchronous tool that requires approval before execution.
func ApprovalTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	approvalFn ApprovalFunc,
	opts ...ChannelToolOption,
) (Tool, ToolHandler, error) {
	return agentictool.ApprovalTool(name, description, handler, approvalFn, opts...)
}

// MustApprovalTool is like ApprovalTool but panics on definition errors.
func MustApprovalTool[TInput any, TOutput any](
	name, description string,
	handler func(ctx context.Context, input TInput) (TOutput, error),
	approvalFn ApprovalFunc,
	opts ...ChannelToolOption,
) (Tool, ToolHandler) {
	return agentictool.MustApprovalTool(name, description, handler, approvalFn, opts...)
}

// WithApproval configures approval on a channel-backed tool.
func WithApproval(fn ApprovalFunc) ChannelToolOption {
	return agentictool.WithApproval(fn)
}

// WithChannelTimeout sets how long a channel-backed tool waits for a result.
func WithChannelTimeout(d time.Duration) ChannelToolOption {
	return agentictool.WithChannelTimeout(d)
}
