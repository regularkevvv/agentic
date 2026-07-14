package agentic_test

import (
	"os"
	"path/filepath"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestNewImageFileMessageAdditionalMediaTypes(t *testing.T) {
	dir := t.TempDir()
	tests := []struct {
		name      string
		extension string
		mediaType string
	}{
		{name: "bmp", extension: ".bmp", mediaType: "image/bmp"},
		{name: "svg", extension: ".svg", mediaType: "image/svg+xml"},
		{name: "pdf", extension: ".pdf", mediaType: "application/pdf"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(dir, tt.name+tt.extension)
			if err := os.WriteFile(path, []byte("data"), 0644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}

			msg, err := agentic.NewImageFileMessage("inspect", path)
			if err != nil {
				t.Fatalf("NewImageFileMessage: %v", err)
			}
			if got := msg.Content[1].ImageData.MediaType; got != tt.mediaType {
				t.Fatalf("expected media type %q, got %q", tt.mediaType, got)
			}
		})
	}
}
