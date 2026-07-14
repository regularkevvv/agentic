package tool

import "testing"

func TestWithToolMaxRetriesAndApplyToolOptions(t *testing.T) {
	if cfg := applyToolOptions(nil); cfg != nil {
		t.Fatalf("expected nil config when no options are provided, got %#v", cfg)
	}

	cfg := applyToolOptions([]ToolOption{WithToolMaxRetries(3)})
	if cfg == nil || cfg.MaxRetries == nil || *cfg.MaxRetries != 3 {
		t.Fatalf("expected max retries to be set to 3, got %#v", cfg)
	}
}
