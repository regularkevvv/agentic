package agentic

import "testing"

func TestInferMediaTypeAdditionalExtensions(t *testing.T) {
	tests := map[string]string{
		"poster.gif":   "image/gif",
		"texture.webp": "image/webp",
		"photo.jpg":    "image/jpeg",
		"photo.jpeg":   "image/jpeg",
	}

	for path, want := range tests {
		if got := inferMediaType(path); got != want {
			t.Fatalf("inferMediaType(%q) = %q, want %q", path, got, want)
		}
	}
}
