package agentic_test

import (
	"os"
	"path/filepath"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestTextPart(t *testing.T) {
	p := agentic.TextPart("hello")
	if p.Type != agentic.ContentText || p.Text != "hello" {
		t.Errorf("unexpected text part: %+v", p)
	}
}

func TestImageURLPart(t *testing.T) {
	p := agentic.ImageURLPart("https://example.com/img.png")
	if p.Type != agentic.ContentImageURL || p.ImageURL == nil {
		t.Fatalf("unexpected part type")
	}
	if p.ImageURL.URL != "https://example.com/img.png" {
		t.Errorf("URL mismatch")
	}
	if p.ImageURL.Detail != "" {
		t.Errorf("expected no detail, got %q", p.ImageURL.Detail)
	}

	// With detail
	p2 := agentic.ImageURLPart("https://example.com/img.png", "high")
	if p2.ImageURL.Detail != "high" {
		t.Errorf("expected detail 'high', got %q", p2.ImageURL.Detail)
	}
}

func TestImageDataPart(t *testing.T) {
	data := []byte("fake image data")
	p := agentic.ImageDataPart(data, "image/jpeg")
	if p.Type != agentic.ContentImageData || p.ImageData == nil {
		t.Fatalf("unexpected part type")
	}
	if p.ImageData.MediaType != "image/jpeg" {
		t.Errorf("media type mismatch")
	}
	if p.ImageData.Data == "" {
		t.Error("expected base64-encoded data")
	}
}

func TestAudioURLPart(t *testing.T) {
	p := agentic.AudioURLPart("https://example.com/audio.mp3", "mp3")
	if p.Type != agentic.ContentAudioURL || p.AudioURL == nil {
		t.Fatalf("unexpected part type")
	}
	if p.AudioURL.URL != "https://example.com/audio.mp3" {
		t.Errorf("URL mismatch")
	}
	if p.AudioURL.Format != "mp3" {
		t.Errorf("format mismatch")
	}
}

func TestVideoURLPart(t *testing.T) {
	p := agentic.VideoURLPart("https://example.com/video.mp4")
	if p.Type != agentic.ContentVideoURL || p.VideoURL == nil {
		t.Fatalf("unexpected part type")
	}
	if p.VideoURL.URL != "https://example.com/video.mp4" {
		t.Errorf("URL mismatch")
	}
}

func TestDocumentURLPart(t *testing.T) {
	p := agentic.DocumentURLPart("https://example.com/doc.pdf", "application/pdf")
	if p.Type != agentic.ContentDocumentURL || p.DocumentURL == nil {
		t.Fatalf("unexpected part type")
	}
	if p.DocumentURL.URL != "https://example.com/doc.pdf" {
		t.Errorf("URL mismatch")
	}
	if p.DocumentURL.MediaType != "application/pdf" {
		t.Errorf("media type mismatch")
	}
}

func TestNewMultiPartMessage(t *testing.T) {
	msg := agentic.NewMultiPartMessage(
		agentic.TextPart("Look at this image"),
		agentic.ImageURLPart("https://example.com/img.png"),
	)
	if msg.Role != agentic.RoleUser {
		t.Errorf("expected user role, got %s", msg.Role)
	}
	if len(msg.Content) != 2 {
		t.Errorf("expected 2 parts, got %d", len(msg.Content))
	}
}

func TestNewImageURLMessage(t *testing.T) {
	msg := agentic.NewImageURLMessage("Describe this", "https://example.com/img.png", "low")
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != agentic.ContentText {
		t.Errorf("first part should be text")
	}
	if msg.Content[1].Type != agentic.ContentImageURL {
		t.Errorf("second part should be image_url")
	}
	if msg.Content[1].ImageURL.Detail != "low" {
		t.Errorf("expected detail 'low'")
	}
}

func TestNewImageDataMessage(t *testing.T) {
	msg := agentic.NewImageDataMessage("What's in this?", []byte{0x89, 0x50}, "image/png")
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(msg.Content))
	}
	if msg.Content[1].Type != agentic.ContentImageData {
		t.Errorf("second part should be image_data")
	}
}

func TestNewImageFileMessage(t *testing.T) {
	// Create a temp image file
	dir := t.TempDir()
	imgPath := filepath.Join(dir, "test.png")
	if err := os.WriteFile(imgPath, []byte{0x89, 0x50, 0x4e, 0x47}, 0644); err != nil {
		t.Fatalf("create temp file: %v", err)
	}

	msg, err := agentic.NewImageFileMessage("What's this?", imgPath)
	if err != nil {
		t.Fatalf("NewImageFileMessage: %v", err)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(msg.Content))
	}
	if msg.Content[1].ImageData.MediaType != "image/png" {
		t.Errorf("expected image/png, got %s", msg.Content[1].ImageData.MediaType)
	}
}

func TestNewImageFileMessage_NotFound(t *testing.T) {
	_, err := agentic.NewImageFileMessage("test", "/nonexistent/path.png")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestNewImageFileMessage_UnknownExtension(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.xyz")
	os.WriteFile(path, []byte("data"), 0644)

	_, err := agentic.NewImageFileMessage("test", path)
	if err == nil {
		t.Error("expected error for unknown extension")
	}
}

func TestCachePointPart(t *testing.T) {
	// Default TTL
	p := agentic.CachePointPart()
	if p.Type != agentic.ContentCachePoint || p.CachePoint == nil {
		t.Fatalf("unexpected part type")
	}
	if p.CachePoint.TTL != "5m" {
		t.Errorf("expected default TTL '5m', got %q", p.CachePoint.TTL)
	}

	// Custom TTL
	p2 := agentic.CachePointPart("1h")
	if p2.CachePoint.TTL != "1h" {
		t.Errorf("expected TTL '1h', got %q", p2.CachePoint.TTL)
	}
}

func TestCachePointInMessage(t *testing.T) {
	// CachePoint can be used in multi-part messages
	msg := agentic.NewMultiPartMessage(
		agentic.TextPart("Context to cache"),
		agentic.CachePointPart(),
		agentic.TextPart("Question after cache point"),
	)

	if len(msg.Content) != 3 {
		t.Fatalf("expected 3 parts, got %d", len(msg.Content))
	}
	if msg.Content[1].Type != agentic.ContentCachePoint {
		t.Error("second part should be cache_point")
	}
}

func TestUploadedFilePart(t *testing.T) {
	p := agentic.UploadedFilePart("file-abc123", "openai")
	if p.Type != agentic.ContentUploadedFile || p.UploadedFile == nil {
		t.Fatalf("unexpected part type")
	}
	if p.UploadedFile.FileID != "file-abc123" {
		t.Errorf("file ID mismatch")
	}
	if p.UploadedFile.ProviderName != "openai" {
		t.Errorf("provider name mismatch")
	}
}

func TestUploadedFileInMessage(t *testing.T) {
	msg := agentic.NewMultiPartMessage(
		agentic.TextPart("Analyze this file"),
		agentic.UploadedFilePart("file-xyz", "anthropic"),
	)

	if len(msg.Content) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(msg.Content))
	}
	if msg.Content[1].UploadedFile.FileID != "file-xyz" {
		t.Error("uploaded file not preserved")
	}
}

func TestVendorMetadata(t *testing.T) {
	// ImageData with vendor_metadata
	p := agentic.Part{
		Type: agentic.ContentImageData,
		ImageData: &agentic.ImageData{
			Data:      "base64data",
			MediaType: "image/png",
			VendorMetadata: map[string]interface{}{
				"detail": "high",
			},
		},
	}
	if p.ImageData.VendorMetadata["detail"] != "high" {
		t.Error("vendor metadata not preserved on ImageData")
	}

	// VideoURL with vendor_metadata
	p2 := agentic.Part{
		Type: agentic.ContentVideoURL,
		VideoURL: &agentic.VideoURL{
			URL: "https://example.com/video.mp4",
			VendorMetadata: map[string]interface{}{
				"processing": "fast",
			},
		},
	}
	if p2.VideoURL.VendorMetadata["processing"] != "fast" {
		t.Error("vendor metadata not preserved on VideoURL")
	}
}
