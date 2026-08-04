package core

import "testing"

func TestChatRequestValidate(t *testing.T) {
	tests := []struct {
		name    string
		req     ChatRequest
		wantErr bool
	}{
		{
			name:    "empty model",
			req:     ChatRequest{Model: "", Messages: []Message{{Role: RoleUser}}},
			wantErr: true,
		},
		{
			name:    "empty messages",
			req:     ChatRequest{Model: "gpt-4", Messages: nil},
			wantErr: true,
		},
		{
			name: "temperature too low",
			req: ChatRequest{
				Model:       "gpt-4",
				Messages:    []Message{{Role: RoleUser}},
				Temperature: floatPtr(-0.1),
			},
			wantErr: true,
		},
		{
			name: "temperature too high",
			req: ChatRequest{
				Model:       "gpt-4",
				Messages:    []Message{{Role: RoleUser}},
				Temperature: floatPtr(2.1),
			},
			wantErr: true,
		},
		{
			name: "negative max tokens",
			req: ChatRequest{
				Model:     "gpt-4",
				Messages:  []Message{{Role: RoleUser}},
				MaxTokens: intPtr(-1),
			},
			wantErr: true,
		},
		{
			name: "valid request",
			req: ChatRequest{
				Model:       "gpt-4",
				Messages:    []Message{{Role: RoleUser}},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
			},
			wantErr: false,
		},
		{
			name: "valid request minimal",
			req: ChatRequest{
				Model:    "gpt-4",
				Messages: []Message{{Role: RoleUser}},
			},
			wantErr: false,
		},
		{
			name:    "invalid prompt cache retention",
			req:     ChatRequest{Model: "gpt-4", Messages: []Message{{Role: RoleUser}}, PromptCache: &PromptCacheConfig{Key: "x", Retention: "forever"}},
			wantErr: true,
		},
		{
			name:    "enabled prompt cache needs key",
			req:     ChatRequest{Model: "gpt-4", Messages: []Message{{Role: RoleUser}}, PromptCache: &PromptCacheConfig{Retention: PromptCacheShort}},
			wantErr: true,
		},
		{
			name:    "disabled prompt cache may omit key",
			req:     ChatRequest{Model: "gpt-4", Messages: []Message{{Role: RoleUser}}, PromptCache: &PromptCacheConfig{Retention: PromptCacheNone}},
			wantErr: false,
		},
		{
			name:    "empty retention with key remains provider opt out",
			req:     ChatRequest{Model: "gpt-4", Messages: []Message{{Role: RoleUser}}, PromptCache: &PromptCacheConfig{Key: "x"}},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.req.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPromptCacheConfigSemantics(t *testing.T) {
	t.Parallel()
	var absent *PromptCacheConfig
	if absent.Enabled() || absent.TTL() != "" {
		t.Fatal("nil cache config enabled")
	}
	if (&PromptCacheConfig{Key: "x", Retention: PromptCacheNone}).Enabled() {
		t.Fatal("none cache enabled")
	}
	short := &PromptCacheConfig{Key: "x", Retention: PromptCacheShort}
	long := &PromptCacheConfig{Key: "x", Retention: PromptCacheLong}
	if !short.Enabled() || short.TTL() != "5m" || !long.Enabled() || long.TTL() != "1h" {
		t.Fatalf("short=%v/%q long=%v/%q", short.Enabled(), short.TTL(), long.Enabled(), long.TTL())
	}
}

func TestUsageAdd(t *testing.T) {
	u := Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	u.Add(Usage{PromptTokens: 20, CompletionTokens: 10, TotalTokens: 30})

	if u.PromptTokens != 30 {
		t.Errorf("expected 30 prompt tokens, got %d", u.PromptTokens)
	}
	if u.CompletionTokens != 15 {
		t.Errorf("expected 15 completion tokens, got %d", u.CompletionTokens)
	}
	if u.TotalTokens != 45 {
		t.Errorf("expected 45 total tokens, got %d", u.TotalTokens)
	}
}

func floatPtr(f float64) *float64 { return &f }
func intPtr(i int) *int           { return &i }
