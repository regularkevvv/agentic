package core

import (
	"strings"
	"testing"
)

func TestFormatToolResult(t *testing.T) {
	if got := FormatToolResult(nil); got != "" {
		t.Fatalf("expected empty string for nil result, got %q", got)
	}

	if got := FormatToolResult("plain text"); got != "plain text" {
		t.Fatalf("expected raw string, got %q", got)
	}

	if got := FormatToolResult(map[string]any{"ok": true}); !strings.Contains(got, `"ok":true`) {
		t.Fatalf("expected JSON output, got %q", got)
	}

	if got := FormatToolResult(make(chan int)); !strings.Contains(got, "Error formatting result") {
		t.Fatalf("expected formatting error message, got %q", got)
	}
}
