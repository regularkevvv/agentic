package gemini

import (
	"testing"

	"github.com/regularkevvv/agentic/internal/core"
)

// Gemini declares a MIME type alongside a file URI rather than sniffing it, so
// announcing a PNG as image/jpeg misdescribes the bytes to the model. The
// provider used to hardcode the defaults; these cover both the supplied and
// the fallback path for images and video.
func TestBuildRequestUsesDeclaredMediaTypes(t *testing.T) {
	tests := []struct {
		name string
		part core.Part
		want string
	}{
		{
			name: "image media type is honored",
			part: core.Part{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://example.com/i.png", MediaType: "image/png"}},
			want: "image/png",
		},
		{
			name: "image falls back to jpeg",
			part: core.Part{Type: core.ContentImageURL, ImageURL: &core.ImageURL{URL: "https://example.com/i.jpg"}},
			want: "image/jpeg",
		},
		{
			name: "video media type is honored",
			part: core.Part{Type: core.ContentVideoURL, VideoURL: &core.VideoURL{URL: "https://example.com/v.webm", MediaType: "video/webm"}},
			want: "video/webm",
		},
		{
			name: "video falls back to mp4",
			part: core.Part{Type: core.ContentVideoURL, VideoURL: &core.VideoURL{URL: "https://example.com/v.mp4"}},
			want: "video/mp4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &Model{model: "gemini-2.5-pro"}
			contents, _ := model.buildRequest(&core.ChatRequest{
				Model:    "gemini-2.5-pro",
				Messages: []core.Message{{Role: core.RoleUser, Content: []core.Part{tt.part}}},
			})

			if len(contents) != 1 || len(contents[0].Parts) != 1 {
				t.Fatalf("expected a single part, got %#v", contents)
			}
			fd := contents[0].Parts[0].FileData
			if fd == nil {
				t.Fatalf("expected file data, got %#v", contents[0].Parts[0])
			}
			if fd.MIMEType != tt.want {
				t.Errorf("MIME type = %q, want %q", fd.MIMEType, tt.want)
			}
		})
	}
}
