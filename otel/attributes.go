package agenticotel

import (
	"context"
	"errors"
	"reflect"
	"sort"

	"github.com/regularkevvv/agentic"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const (
	keyOperation        = "gen_ai.operation.name"
	keyProvider         = "gen_ai.provider.name"
	keyRequestModel     = "gen_ai.request.model"
	keyResponseModel    = "gen_ai.response.model"
	keyResponseID       = "gen_ai.response.id"
	keyFinishReasons    = "gen_ai.response.finish_reasons"
	keyInputTokens      = "gen_ai.usage.input_tokens"
	keyOutputTokens     = "gen_ai.usage.output_tokens"
	keyCacheRead        = "gen_ai.usage.cache_read.input_tokens"
	keyCacheCreation    = "gen_ai.usage.cache_creation.input_tokens"
	keyReasoningTokens  = "gen_ai.usage.reasoning.output_tokens"
	keyConversationID   = "gen_ai.conversation.id"
	keyAgentName        = "gen_ai.agent.name"
	keyAgentDescription = "gen_ai.agent.description"
	keyToolName         = "gen_ai.tool.name"
	keyToolType         = "gen_ai.tool.type"
	keyToolCallID       = "gen_ai.tool.call.id"
	keyErrorType        = "error.type"
	keyServerAddress    = "server.address"
	keyServerPort       = "server.port"
	keyInputMessages    = "gen_ai.input.messages"
	keyOutputMessages   = "gen_ai.output.messages"
	keySystem           = "gen_ai.system_instructions"
	keyToolDefinitions  = "gen_ai.tool.definitions"
	keyToolArguments    = "gen_ai.tool.call.arguments"
	keyToolResult       = "gen_ai.tool.call.result"
	keyTTFC             = "gen_ai.response.time_to_first_chunk"
	keyRunID            = "agentic.run.id"
	keyAgentVersion     = "agentic.agent.version"
	keyExecutionMode    = "agentic.execution.mode"
	keyExecutionOutcome = "agentic.execution.outcome"
	keyToolAttempt      = "agentic.tool.attempt"
	keyToolResumed      = "agentic.tool.handler_resumed"
	keyToolOutcome      = "agentic.tool.outcome"
)

func stringAttr(key, value string, attrs *[]attribute.KeyValue) {
	if value != "" {
		*attrs = append(*attrs, attribute.String(key, value))
	}
}

func invokedAgentAttrs(identity agentic.AgentIdentity) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 2)
	stringAttr(keyAgentName, identity.Name, &attrs)
	stringAttr(keyAgentDescription, identity.Description, &attrs)
	stringAttr(keyAgentVersion, identity.Version, &attrs)
	return attrs
}

func agentContextAttrs(identity agentic.AgentIdentity) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 1)
	stringAttr(keyAgentName, identity.Name, &attrs)
	return attrs
}

func runAttrs(run agentic.RunMetadata) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 2)
	stringAttr(keyConversationID, run.ConversationID, &attrs)
	stringAttr(keyRunID, run.RunID, &attrs)
	return attrs
}

func modelAttrs(metadata agentic.ModelMetadata, model string) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 5)
	stringAttr(keyOperation, metadata.Operation, &attrs)
	stringAttr(keyProvider, metadata.Provider, &attrs)
	stringAttr(keyRequestModel, model, &attrs)
	stringAttr(keyServerAddress, metadata.ServerAddress, &attrs)
	if metadata.ServerPort > 0 {
		attrs = append(attrs, attribute.Int(keyServerPort, metadata.ServerPort))
	}
	return attrs
}

func requestAttrs(request agentic.ChatRequest) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 7)
	if request.MaxTokens != nil {
		attrs = append(attrs, attribute.Int("gen_ai.request.max_tokens", *request.MaxTokens))
	}
	if request.Temperature != nil {
		attrs = append(attrs, attribute.Float64("gen_ai.request.temperature", *request.Temperature))
	}
	if request.TopP != nil {
		attrs = append(attrs, attribute.Float64("gen_ai.request.top_p", *request.TopP))
	}
	if len(request.StopSequences) > 0 {
		attrs = append(attrs, attribute.StringSlice("gen_ai.request.stop_sequences", request.StopSequences))
	}
	if request.Stream {
		attrs = append(attrs, attribute.Bool("gen_ai.request.stream", true))
	}
	if request.ResponseFormat != nil {
		attrs = append(attrs, attribute.String("gen_ai.output.type", "json"))
	}
	return attrs
}

func agentRequestAttrs(request agentic.ChatRequest) []attribute.KeyValue {
	attrs := requestAttrs(request)
	if !request.Stream {
		return attrs
	}
	// gen_ai.request.stream belongs to inference client spans, not the
	// in-process invoke-agent convention.
	filtered := attrs[:0]
	for _, attr := range attrs {
		if string(attr.Key) != "gen_ai.request.stream" {
			filtered = append(filtered, attr)
		}
	}
	return filtered
}

func responseAttrs(response *agentic.ChatResponse) []attribute.KeyValue {
	if response == nil {
		return nil
	}
	attrs := make([]attribute.KeyValue, 0, 8)
	stringAttr(keyResponseID, response.ID, &attrs)
	stringAttr(keyResponseModel, response.Model, &attrs)
	if response.FinishReason != "" {
		attrs = append(attrs, attribute.StringSlice(keyFinishReasons, []string{string(response.FinishReason)}))
	}
	if response.Usage.PromptTokens > 0 {
		attrs = append(attrs, attribute.Int(keyInputTokens, response.Usage.PromptTokens))
	}
	if response.Usage.CompletionTokens > 0 {
		attrs = append(attrs, attribute.Int(keyOutputTokens, response.Usage.CompletionTokens))
	}
	if response.Usage.CacheReadTokens > 0 {
		attrs = append(attrs, attribute.Int(keyCacheRead, response.Usage.CacheReadTokens))
	}
	if response.Usage.CacheCreationTokens > 0 {
		attrs = append(attrs, attribute.Int(keyCacheCreation, response.Usage.CacheCreationTokens))
	}
	if response.Usage.ReasoningTokens > 0 {
		attrs = append(attrs, attribute.Int(keyReasoningTokens, response.Usage.ReasoningTokens))
	}
	return attrs
}

func errorType(err error) string {
	if err == nil {
		return ""
	}
	if errors.Is(err, context.Canceled) {
		return "canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "deadline_exceeded"
	}
	for {
		var next error
		if joined, ok := err.(interface{ Unwrap() []error }); ok {
			for _, candidate := range joined.Unwrap() {
				if candidate != nil {
					next = candidate
					break
				}
			}
		} else {
			next = errors.Unwrap(err)
		}
		if next == nil {
			break
		}
		err = next
	}
	typ := reflect.TypeOf(err)
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	if typ.PkgPath() == "" {
		return typ.Name()
	}
	return typ.PkgPath() + "." + typ.Name()
}

func markSpanError(span trace.Span, err error, includeContent bool) string {
	kind := errorType(err)
	if kind == "" {
		return ""
	}
	span.SetAttributes(attribute.String(keyErrorType, kind))
	span.SetStatus(codes.Error, kind)
	if includeContent {
		span.RecordError(err)
	}
	return kind
}

func valueFromAny(value any) attribute.Value {
	switch value := value.(type) {
	case nil:
		return attribute.Value{}
	case string:
		return attribute.StringValue(value)
	case bool:
		return attribute.BoolValue(value)
	case int:
		return attribute.IntValue(value)
	case int64:
		return attribute.Int64Value(value)
	case float64:
		return attribute.Float64Value(value)
	case float32:
		return attribute.Float64Value(float64(value))
	case []any:
		values := make([]attribute.Value, 0, len(value))
		for _, item := range value {
			values = append(values, valueFromAny(item))
		}
		return attribute.SliceValue(values...)
	case map[string]any:
		keys := make([]string, 0, len(value))
		for key := range value {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		values := make([]attribute.KeyValue, 0, len(keys))
		for _, key := range keys {
			values = append(values, attribute.KeyValue{Key: attribute.Key(key), Value: valueFromAny(value[key])})
		}
		return attribute.MapValue(values...)
	default:
		return attribute.StringValue(reflect.ValueOf(value).String())
	}
}
