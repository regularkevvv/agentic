package agenticotel

import (
	"encoding/json"
	"strings"

	"github.com/regularkevvv/agentic"

	"go.opentelemetry.io/otel/attribute"
)

func (i *Instrumentation) filtered(kind ContentKind, value string) (filtered string, ok bool) {
	if i.config.filter == nil {
		return value, true
	}
	defer func() {
		if recover() != nil {
			filtered, ok = "", false
		}
	}()
	return i.config.filter(kind, value)
}

func (i *Instrumentation) contentAttribute(key string, value any) (attribute.KeyValue, bool) {
	encoded, err := json.Marshal(value)
	if err != nil || (i.config.maxContentBytes > 0 && len(encoded) > i.config.maxContentBytes) {
		return attribute.KeyValue{}, false
	}
	var normalized any
	if json.Unmarshal(encoded, &normalized) != nil {
		return attribute.KeyValue{}, false
	}
	if normalized == nil {
		return attribute.KeyValue{}, false
	}
	return attribute.KeyValue{Key: attribute.Key(key), Value: valueFromAny(normalized)}, true
}

func (i *Instrumentation) messageContent(messages []agentic.Message) (system, regular []any) {
	for _, message := range messages {
		parts := i.messageParts(message)
		if len(parts) == 0 {
			continue
		}
		if message.Role == agentic.RoleSystem {
			system = append(system, parts...)
			continue
		}
		regular = append(regular, map[string]any{"role": string(message.Role), "parts": parts})
	}
	return system, regular
}

func (i *Instrumentation) outputMessageContent(messages []agentic.Message, finishReason agentic.FinishReason) []any {
	if finishReason == "" {
		return nil
	}
	// Agentic represents one provider candidate per ChatResponse. An agent
	// invocation may accumulate intermediate assistant/tool messages, but the
	// invoke-agent output is its final assistant response. Emitting only that
	// response also ensures every item follows the output-message schema, which
	// requires a finish_reason.
	for index := len(messages) - 1; index >= 0; index-- {
		message := messages[index]
		if message.Role != agentic.RoleAssistant {
			continue
		}
		parts := i.messageParts(message)
		if len(parts) == 0 {
			continue
		}
		return []any{map[string]any{
			"role":          string(message.Role),
			"parts":         parts,
			"finish_reason": normalizedFinishReason(finishReason),
		}}
	}
	return nil
}

func (i *Instrumentation) messageParts(message agentic.Message) []any {
	parts := make([]any, 0, len(message.Content))
	for _, part := range message.Content {
		if converted, ok := i.messagePart(part); ok {
			parts = append(parts, converted)
		}
	}
	return parts
}

func normalizedFinishReason(reason agentic.FinishReason) string {
	if reason == agentic.FinishReasonToolCalls {
		return "tool_call"
	}
	return string(reason)
}

func (i *Instrumentation) messagePart(part agentic.Part) (any, bool) {
	switch part.Type {
	case agentic.ContentText:
		text, ok := i.filtered(ContentMessageText, part.Text)
		return map[string]any{"type": "text", "content": text}, ok
	case agentic.ContentThinking:
		if part.Thinking == nil || part.Thinking.IsRedacted() {
			return nil, false
		}
		text, ok := i.filtered(ContentReasoning, part.Thinking.Text)
		return map[string]any{"type": "reasoning", "content": text}, ok
	case agentic.ContentToolUse:
		if !i.config.toolContent || part.ToolUse == nil {
			return nil, false
		}
		arguments, ok := i.filteredJSONObject(ContentToolArguments, part.ToolUse.Input)
		if !ok {
			return nil, false
		}
		return map[string]any{"type": "tool_call", "id": part.ToolUse.ID, "name": part.ToolUse.Name, "arguments": arguments}, true
	case agentic.ContentToolResult:
		if !i.config.toolContent || part.ToolResult == nil {
			return nil, false
		}
		result, ok := i.filtered(ContentToolResult, part.ToolResult.Content)
		if !ok {
			return nil, false
		}
		return map[string]any{"type": "tool_call_response", "id": part.ToolResult.ToolUseID, "response": result}, true
	case agentic.ContentImageURL:
		if part.ImageURL == nil {
			return nil, false
		}
		return i.uriPart("image", part.ImageURL.URL, part.ImageURL.MediaType)
	case agentic.ContentAudioURL:
		if part.AudioURL == nil {
			return nil, false
		}
		return i.uriPart("audio", part.AudioURL.URL, part.AudioURL.Format)
	case agentic.ContentVideoURL:
		if part.VideoURL == nil {
			return nil, false
		}
		return i.uriPart("video", part.VideoURL.URL, part.VideoURL.MediaType)
	case agentic.ContentDocumentURL:
		if part.DocumentURL == nil {
			return nil, false
		}
		return i.uriPart("document", part.DocumentURL.URL, part.DocumentURL.MediaType)
	case agentic.ContentUploadedFile:
		if part.UploadedFile == nil {
			return nil, false
		}
		fileID, ok := i.filtered(ContentFileID, part.UploadedFile.FileID)
		if !ok {
			return nil, false
		}
		return map[string]any{"type": "file", "modality": "document", "file_id": fileID}, true
	default:
		// Inline binary data, provider metadata, and cache markers are never
		// exported by this adapter.
		return nil, false
	}
}

func (i *Instrumentation) uriPart(modality, uri, mime string) (any, bool) {
	// URL-shaped Agentic parts are allowed to contain ordinary remote URIs, but
	// a data URI is inline binary content and is never exported by this adapter.
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(uri)), "data:") {
		return nil, false
	}
	uri, ok := i.filtered(ContentURI, uri)
	if !ok {
		return nil, false
	}
	part := map[string]any{"type": "uri", "modality": modality, "uri": uri}
	if mime != "" && strings.Contains(mime, "/") {
		part["mime_type"] = mime
	}
	return part, true
}

func (i *Instrumentation) filteredJSONObject(kind ContentKind, value any) (map[string]any, bool) {
	var decoded any
	switch value := value.(type) {
	case string:
		if json.Unmarshal([]byte(value), &decoded) != nil {
			return nil, false
		}
	case json.RawMessage:
		if json.Unmarshal(value, &decoded) != nil {
			return nil, false
		}
	case []byte:
		if json.Unmarshal(value, &decoded) != nil {
			return nil, false
		}
	default:
		encoded, err := json.Marshal(value)
		if err != nil || json.Unmarshal(encoded, &decoded) != nil {
			return nil, false
		}
	}
	object, ok := decoded.(map[string]any)
	if !ok || object == nil {
		return nil, false
	}
	encoded, err := json.Marshal(object)
	if err != nil {
		return nil, false
	}
	filtered, ok := i.filtered(kind, string(encoded))
	if !ok {
		return nil, false
	}
	var filteredObject map[string]any
	if json.Unmarshal([]byte(filtered), &filteredObject) != nil || filteredObject == nil {
		return nil, false
	}
	return filteredObject, true
}

func (i *Instrumentation) toolDefinitions(tools []agentic.Tool) []any {
	definitions := make([]any, 0, len(tools))
	for _, tool := range tools {
		definition := map[string]any{"type": string(tool.Type), "name": tool.Function.Name}
		if description, ok := i.filtered(ContentToolDescription, tool.Function.Description); ok && description != "" {
			definition["description"] = description
		}
		if len(tool.Function.Parameters) > 0 {
			if parameters, ok := i.filteredJSONObject(ContentToolParameters, tool.Function.Parameters); ok {
				definition["parameters"] = parameters
			}
		}
		definitions = append(definitions, definition)
	}
	return definitions
}
