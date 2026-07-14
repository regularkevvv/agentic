package agentic

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// --- Part constructors ---

// TextPart creates a text content part.
func TextPart(text string) Part {
	return Part{Type: ContentText, Text: text}
}

// ImageURLPart creates an image-from-URL content part.
// Optional detail parameter controls image processing fidelity ("auto", "low", "high").
func ImageURLPart(url string, detail ...string) Part {
	img := &ImageURL{URL: url}
	if len(detail) > 0 {
		img.Detail = detail[0]
	}
	return Part{Type: ContentImageURL, ImageURL: img}
}

// ImageDataPart creates an inline base64-encoded image content part.
func ImageDataPart(data []byte, mediaType string) Part {
	return Part{
		Type: ContentImageData,
		ImageData: &ImageData{
			Data:      base64.StdEncoding.EncodeToString(data),
			MediaType: mediaType,
		},
	}
}

// AudioURLPart creates an audio-from-URL content part.
func AudioURLPart(url string, format ...string) Part {
	audio := &AudioURL{URL: url}
	if len(format) > 0 {
		audio.Format = format[0]
	}
	return Part{Type: ContentAudioURL, AudioURL: audio}
}

// VideoURLPart creates a video-from-URL content part.
func VideoURLPart(url string) Part {
	return Part{Type: ContentVideoURL, VideoURL: &VideoURL{URL: url}}
}

// DocumentURLPart creates a document-from-URL content part (e.g. PDF).
func DocumentURLPart(url string, mediaType ...string) Part {
	doc := &DocumentURL{URL: url}
	if len(mediaType) > 0 {
		doc.MediaType = mediaType[0]
	}
	return Part{Type: ContentDocumentURL, DocumentURL: doc}
}

// CachePointPart creates a cache point marker.
// When inserted into message content, it signals the provider to cache
// all content preceding this point. TTL can be "5m" (default) or "1h".
// Supported by: Anthropic, Amazon Bedrock. Silently ignored by other providers.
func CachePointPart(ttl ...string) Part {
	cp := &CachePoint{TTL: "5m"}
	if len(ttl) > 0 {
		cp.TTL = ttl[0]
	}
	return Part{Type: ContentCachePoint, CachePoint: cp}
}

// UploadedFilePart creates a reference to a file uploaded to a provider's file storage.
// The fileID is the provider-specific identifier returned by the provider's upload API.
// The providerName identifies which provider hosts the file (e.g. "anthropic", "openai").
func UploadedFilePart(fileID string, providerName string) Part {
	return Part{
		Type: ContentUploadedFile,
		UploadedFile: &UploadedFile{
			FileID:       fileID,
			ProviderName: providerName,
		},
	}
}

// --- Message constructors ---

// NewMultiPartMessage creates a user message with multiple content parts.
func NewMultiPartMessage(parts ...Part) Message {
	return Message{
		Role:    RoleUser,
		Content: parts,
	}
}

// NewImageURLMessage creates a user message with text and an image URL.
func NewImageURLMessage(text string, imageURL string, detail ...string) Message {
	parts := []Part{TextPart(text), ImageURLPart(imageURL, detail...)}
	return NewMultiPartMessage(parts...)
}

// NewImageDataMessage creates a user message with text and inline image data.
func NewImageDataMessage(text string, data []byte, mediaType string) Message {
	parts := []Part{TextPart(text), ImageDataPart(data, mediaType)}
	return NewMultiPartMessage(parts...)
}

// NewImageFileMessage creates a user message with text and an image loaded from a file.
// The media type is inferred from the file extension.
func NewImageFileMessage(text string, filePath string) (Message, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return Message{}, fmt.Errorf("read image file: %w", err)
	}

	mediaType := inferMediaType(filePath)
	if mediaType == "" {
		return Message{}, fmt.Errorf("unable to determine media type for %s", filePath)
	}

	return NewImageDataMessage(text, data, mediaType), nil
}

// inferMediaType determines the MIME type from a file extension.
func inferMediaType(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".bmp":
		return "image/bmp"
	case ".svg":
		return "image/svg+xml"
	case ".pdf":
		return "application/pdf"
	default:
		return ""
	}
}
