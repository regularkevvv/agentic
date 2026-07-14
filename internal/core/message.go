package core

// MessageRole represents the role of a message in a conversation.
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleTool      MessageRole = "tool"
)

// ContentType represents the type of content in a message part.
type ContentType string

const (
	ContentText         ContentType = "text"
	ContentImageURL     ContentType = "image_url"
	ContentToolUse      ContentType = "tool_use"
	ContentToolResult   ContentType = "tool_result"
	ContentThinking     ContentType = "thinking"
	ContentImageData    ContentType = "image_data"
	ContentAudioURL     ContentType = "audio_url"
	ContentVideoURL     ContentType = "video_url"
	ContentDocumentURL  ContentType = "document_url"
	ContentCachePoint   ContentType = "cache_point"
	ContentUploadedFile ContentType = "uploaded_file"
)

// Message represents a standardized message in an LLM conversation.
type Message struct {
	Role    MessageRole `json:"role"`
	Content []Part      `json:"content"`
}

// Part represents a part of message content (text, image, tool use, etc.).
type Part struct {
	Type         ContentType    `json:"type"`
	Text         string         `json:"text,omitempty"`
	ImageURL     *ImageURL      `json:"image_url,omitempty"`
	ToolUse      *ToolUse       `json:"tool_use,omitempty"`
	ToolResult   *ToolResult    `json:"tool_result,omitempty"`
	Thinking     *ThinkingBlock `json:"thinking,omitempty"`
	ImageData    *ImageData     `json:"image_data,omitempty"`
	AudioURL     *AudioURL      `json:"audio_url,omitempty"`
	VideoURL     *VideoURL      `json:"video_url,omitempty"`
	DocumentURL  *DocumentURL   `json:"document_url,omitempty"`
	CachePoint   *CachePoint    `json:"cache_point,omitempty"`
	UploadedFile *UploadedFile  `json:"uploaded_file,omitempty"`
}

// ImageURL represents an image referenced by URL in a message.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
}

// ThinkingBlock represents a thinking/reasoning block from the model.
type ThinkingBlock struct {
	// Text is the thinking content. Empty for redacted thinking blocks.
	Text string `json:"text"`
	// ID identifies the thinking block. Set to "redacted_thinking" for redacted blocks.
	ID string `json:"id,omitempty"`
	// Signature is the provider-specific signature (Anthropic, Bedrock, Google, OpenAI).
	// For redacted thinking, this contains the encrypted data.
	Signature string `json:"signature,omitempty"`
	// ProviderName is the name of the provider that generated this block.
	ProviderName string `json:"provider_name,omitempty"`
	// ProviderDetails holds additional provider-specific data.
	ProviderDetails map[string]interface{} `json:"provider_details,omitempty"`
}

// IsRedacted returns true if this is a redacted thinking block.
func (tb *ThinkingBlock) IsRedacted() bool {
	return tb.ID == "redacted_thinking"
}

// ImageData represents an inline image with base64-encoded data.
type ImageData struct {
	Data           string                 `json:"data"`                      // base64-encoded
	MediaType      string                 `json:"media_type"`                // e.g. "image/png", "image/jpeg"
	VendorMetadata map[string]interface{} `json:"vendor_metadata,omitempty"` // Provider-specific metadata (e.g. OpenAI detail level)
}

// AudioURL represents an audio file referenced by URL.
type AudioURL struct {
	URL            string                 `json:"url"`
	Format         string                 `json:"format,omitempty"`          // e.g. "mp3", "wav"
	VendorMetadata map[string]interface{} `json:"vendor_metadata,omitempty"` // Provider-specific metadata
}

// VideoURL represents a video file referenced by URL.
type VideoURL struct {
	URL            string                 `json:"url"`
	VendorMetadata map[string]interface{} `json:"vendor_metadata,omitempty"` // Provider-specific metadata (e.g. Google video_metadata)
}

// DocumentURL represents a document (e.g. PDF) referenced by URL.
type DocumentURL struct {
	URL            string                 `json:"url"`
	MediaType      string                 `json:"media_type,omitempty"`      // e.g. "application/pdf"
	VendorMetadata map[string]interface{} `json:"vendor_metadata,omitempty"` // Provider-specific metadata
}

// CachePoint is a marker for prompt caching boundaries.
// When inserted into message content, it signals the provider to cache
// all content up to this point for faster subsequent requests.
// Supported by: Anthropic, Amazon Bedrock.
type CachePoint struct {
	// TTL is the cache time-to-live: "5m" (5 minutes) or "1h" (1 hour).
	// Default is "5m". Bedrock does not support explicit TTL.
	TTL string `json:"ttl,omitempty"` // "5m" or "1h"
}

// UploadedFile references a file that has been uploaded to a provider's file storage.
// This allows referencing files by ID rather than providing content inline.
type UploadedFile struct {
	// FileID is the provider-specific file identifier.
	FileID string `json:"file_id"`
	// ProviderName identifies which provider hosts the file.
	// Values: "anthropic", "openai", "google-gla", "google-vertex", "bedrock".
	ProviderName string `json:"provider_name"`
	// VendorMetadata holds provider-specific metadata for the file.
	VendorMetadata map[string]interface{} `json:"vendor_metadata,omitempty"`
}

// ToolUse represents a tool being called by the model.
type ToolUse struct {
	ID    string                 `json:"id"`
	Name  string                 `json:"name"`
	Input map[string]interface{} `json:"input"`
}

// ToolResult represents the result of a tool call.
type ToolResult struct {
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
	IsError   bool   `json:"is_error,omitempty"`
}

// NewTextMessage creates a simple text message.
func NewTextMessage(role MessageRole, text string) Message {
	return Message{
		Role: role,
		Content: []Part{
			{
				Type: ContentText,
				Text: text,
			},
		},
	}
}

// NewToolUseMessage creates a message with tool use parts.
func NewToolUseMessage(toolUses ...ToolUse) Message {
	parts := make([]Part, len(toolUses))
	for i, tu := range toolUses {
		tu := tu
		parts[i] = Part{
			Type:    ContentToolUse,
			ToolUse: &tu,
		}
	}
	return Message{
		Role:    RoleAssistant,
		Content: parts,
	}
}

// NewToolResultMessage creates a message with a tool result.
func NewToolResultMessage(toolUseID string, content string, isError bool) Message {
	return Message{
		Role: RoleTool,
		Content: []Part{
			{
				Type: ContentToolResult,
				ToolResult: &ToolResult{
					ToolUseID: toolUseID,
					Content:   content,
					IsError:   isError,
				},
			},
		},
	}
}

// GetTextContent extracts all text content from a message.
func (m *Message) GetTextContent() string {
	var text string
	for _, part := range m.Content {
		if part.Type == ContentText {
			text += part.Text
		}
	}
	return text
}

// GetToolUses extracts all tool uses from a message.
func (m *Message) GetToolUses() []ToolUse {
	var toolUses []ToolUse
	for _, part := range m.Content {
		if part.Type == ContentToolUse && part.ToolUse != nil {
			toolUses = append(toolUses, *part.ToolUse)
		}
	}
	return toolUses
}

// GetToolResults extracts all tool results from a message.
func (m *Message) GetToolResults() []ToolResult {
	var results []ToolResult
	for _, part := range m.Content {
		if part.Type == ContentToolResult && part.ToolResult != nil {
			results = append(results, *part.ToolResult)
		}
	}
	return results
}

// GetThinkingContent extracts all thinking/reasoning text from a message.
func (m *Message) GetThinkingContent() string {
	var text string
	for _, part := range m.Content {
		if part.Type == ContentThinking && part.Thinking != nil {
			text += part.Thinking.Text
		}
	}
	return text
}
