package grok

import (
	"encoding/json"
	"testing"

	"github.com/regularkevvv/agentic/internal/core"

	"github.com/openai/openai-go/packages/respjson"
)

func TestReasoningEfforts(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  map[string]bool
	}{
		{"grok 4.3", "grok-4.3", grok43Efforts},
		{"grok 4.3 latest", "grok-4.3-latest", grok43Efforts},
		{"floating grok alias", "grok-latest", grok43Efforts},
		{"retired slug redirected to 4.3", "grok-3", grok43Efforts},
		{"retired fast slug redirected to 4.3", "grok-4-fast-reasoning", grok43Efforts},
		{"grok 4.5", "grok-4.5", grok45Efforts},
		{"grok 4.5 latest", "grok-4.5-latest", grok45Efforts},
		{"floating build alias", "grok-build-latest", grok45Efforts},
		{"grok 3 mini", "grok-3-mini", basicEfforts},
		{"grok 3 mini variant", "grok-3-mini-fast", basicEfforts},
		{"non-reasoning model", "grok-2-1212", nil},
		{"code model redirected away from 4.3", "grok-code-fast-1", nil},
		{"unknown model", "some-other-model", nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := reasoningEfforts(tt.model)
			if len(got) != len(tt.want) {
				t.Fatalf("reasoningEfforts(%q) = %v, want %v", tt.model, got, tt.want)
			}
			for effort := range tt.want {
				if !got[effort] {
					t.Errorf("reasoningEfforts(%q) missing %q", tt.model, effort)
				}
			}
		})
	}
}

func TestClampReasoningEffort(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		effort   string
		want     string
		wantKeep bool
	}{
		{"supported value passes through", "grok-4.3", "medium", "medium", true},
		{"4.3 accepts none", "grok-4.3", "none", "none", true},
		{"4.5 rejects none so it is dropped", "grok-4.5", "none", "", false},
		{"4.5 accepts medium", "grok-4.5", "medium", "medium", true},
		{"mini has no medium so it rounds up", "grok-3-mini", "medium", "high", true},
		{"mini rejects none so it is dropped", "grok-3-mini", "none", "", false},
		{"minimal maps to low", "grok-3-mini", "minimal", "low", true},
		{"minimal maps to low on 4.5", "grok-4.5", "minimal", "low", true},
		{"non-reasoning model drops every value", "grok-2-1212", "high", "", false},
		{"non-reasoning model drops none", "grok-2-1212", "none", "", false},
		{"unrecognized value is dropped", "grok-4.3", "extreme", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, keep := clampReasoningEffort(tt.model, tt.effort)
			if got != tt.want || keep != tt.wantKeep {
				t.Errorf("clampReasoningEffort(%q, %q) = (%q, %t), want (%q, %t)",
					tt.model, tt.effort, got, keep, tt.want, tt.wantKeep)
			}
		})
	}
}

func TestReasoningEffortFor(t *testing.T) {
	tests := []struct {
		name  string
		model string
		cfg   *core.ThinkingConfig
		want  string
	}{
		{"no thinking config sends nothing", "grok-4.3", nil, ""},
		{"default budget is medium", "grok-4.3", &core.ThinkingConfig{Enabled: true}, "medium"},
		{"large budget is high", "grok-4.3", &core.ThinkingConfig{Enabled: true, BudgetTokens: 30000}, "high"},
		{"small budget is low", "grok-4.3", &core.ThinkingConfig{Enabled: true, BudgetTokens: 4000}, "low"},
		{"disabled thinking sends none on 4.3", "grok-4.3", &core.ThinkingConfig{}, "none"},
		{"disabled thinking sends nothing on 4.5", "grok-4.5", &core.ThinkingConfig{}, ""},
		{"medium is rounded up on mini", "grok-3-mini", &core.ThinkingConfig{Enabled: true}, "high"},
		{"small budget stays low on mini", "grok-3-mini", &core.ThinkingConfig{Enabled: true, BudgetTokens: 1000}, "low"},
		{"non-reasoning model sends nothing", "grok-2-1212", &core.ThinkingConfig{Enabled: true}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := reasoningEffortFor(tt.model, tt.cfg); got != tt.want {
				t.Errorf("reasoningEffortFor(%q, %+v) = %q, want %q", tt.model, tt.cfg, got, tt.want)
			}
		})
	}
}

func TestConvertFinishReason(t *testing.T) {
	tests := []struct {
		reason string
		want   core.FinishReason
	}{
		{"stop", core.FinishReasonStop},
		{"length", core.FinishReasonLength},
		{"max_output_tokens", core.FinishReasonLength},
		{"tool_calls", core.FinishReasonToolCalls},
		{"content_filter", core.FinishReasonContentFilter},
		{"cancelled", core.FinishReasonError},
		{"failed", core.FinishReasonError},
		{"", core.FinishReasonUnknown},
		{"something_new", core.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.reason, func(t *testing.T) {
			if got := convertFinishReason(tt.reason); got != tt.want {
				t.Errorf("convertFinishReason(%q) = %q, want %q", tt.reason, got, tt.want)
			}
		})
	}
}

func TestExtractReasoning(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]respjson.Field
		want  string
	}{
		{"no extra fields", nil, ""},
		{
			name:  "omitted field",
			extra: map[string]respjson.Field{"reasoning": {}},
			want:  "",
		},
		{
			name:  "reasoning field",
			extra: map[string]respjson.Field{"reasoning": respjson.NewInvalidField(`"thought"`)},
			want:  "thought",
		},
		{
			name:  "reasoning_content field",
			extra: map[string]respjson.Field{"reasoning_content": respjson.NewInvalidField(`"thought"`)},
			want:  "thought",
		},
		{
			name: "reasoning wins over reasoning_content",
			extra: map[string]respjson.Field{
				"reasoning":         respjson.NewInvalidField(`"first"`),
				"reasoning_content": respjson.NewInvalidField(`"second"`),
			},
			want: "first",
		},
		{
			name: "falls through an empty reasoning field",
			extra: map[string]respjson.Field{
				"reasoning":         respjson.NewInvalidField(`""`),
				"reasoning_content": respjson.NewInvalidField(`"second"`),
			},
			want: "second",
		},
		{
			name:  "non-string value is ignored",
			extra: map[string]respjson.Field{"reasoning": respjson.NewInvalidField(`{"text":"thought"}`)},
			want:  "",
		},
		{
			name:  "null value is ignored",
			extra: map[string]respjson.Field{"reasoning": respjson.NewInvalidField(`null`)},
			want:  "",
		},
		{
			name:  "unrelated field is ignored",
			extra: map[string]respjson.Field{"citations": respjson.NewInvalidField(`["a"]`)},
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := extractReasoning(tt.extra); got != tt.want {
				t.Errorf("extractReasoning() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestProviderOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    map[string]any
		want    Options
		wantHit bool
	}{
		{"no provider options", nil, Options{}, false},
		{
			name:    "another provider's entry is ignored",
			opts:    map[string]any{"openai": Options{ReasoningEffort: "high"}},
			want:    Options{},
			wantHit: false,
		},
		{
			name:    "value form",
			opts:    map[string]any{ProviderName: Options{ReasoningEffort: "high"}},
			want:    Options{ReasoningEffort: "high"},
			wantHit: true,
		},
		{
			name:    "pointer form",
			opts:    map[string]any{ProviderName: &Options{ReasoningEffort: "low"}},
			want:    Options{ReasoningEffort: "low"},
			wantHit: true,
		},
		{
			name:    "nil pointer is ignored",
			opts:    map[string]any{ProviderName: (*Options)(nil)},
			want:    Options{},
			wantHit: false,
		},
		{
			name:    "mistyped value is ignored",
			opts:    map[string]any{ProviderName: "high"},
			want:    Options{},
			wantHit: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, hit := providerOptions(&core.ChatRequest{ProviderOptions: tt.opts})
			if hit != tt.wantHit || got != tt.want {
				t.Errorf("providerOptions() = (%+v, %t), want (%+v, %t)", got, hit, tt.want, tt.wantHit)
			}
		})
	}
}

func TestResolveReasoningEffort(t *testing.T) {
	tests := []struct {
		name  string
		model string
		req   *core.ChatRequest
		want  string
	}{
		{
			name:  "options override thinking config",
			model: "grok-4.3",
			req: &core.ChatRequest{
				Thinking:        &core.ThinkingConfig{Enabled: true},
				ProviderOptions: map[string]any{ProviderName: Options{ReasoningEffort: "low"}},
			},
			want: "low",
		},
		{
			name:  "override is clamped to what the model accepts",
			model: "grok-3-mini",
			req: &core.ChatRequest{
				ProviderOptions: map[string]any{ProviderName: Options{ReasoningEffort: "medium"}},
			},
			want: "high",
		},
		{
			name:  "unsupported override is dropped",
			model: "grok-4.5",
			req: &core.ChatRequest{
				ProviderOptions: map[string]any{ProviderName: Options{ReasoningEffort: "none"}},
			},
			want: "",
		},
		{
			name:  "empty override falls back to thinking config",
			model: "grok-4.3",
			req: &core.ChatRequest{
				Thinking:        &core.ThinkingConfig{Enabled: true, BudgetTokens: 30000},
				ProviderOptions: map[string]any{ProviderName: Options{}},
			},
			want: "high",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{model: tt.model}
			if got := m.resolveReasoningEffort(tt.req); got != tt.want {
				t.Errorf("resolveReasoningEffort() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestConvertToolChoice(t *testing.T) {
	tests := []struct {
		choice core.ToolChoice
		want   string
	}{
		{core.ToolChoiceNone, `"none"`},
		{core.ToolChoiceAuto, `"auto"`},
		{core.ToolChoiceRequired, `"required"`},
		{core.ToolChoice("nonsense"), `"auto"`},
	}

	for _, tt := range tests {
		t.Run(string(tt.choice), func(t *testing.T) {
			got, err := json.Marshal(convertToolChoice(tt.choice))
			if err != nil {
				t.Fatalf("marshaling tool choice: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("convertToolChoice(%q) = %s, want %s", tt.choice, got, tt.want)
			}
		})
	}
}

func TestConvertResponseFormat(t *testing.T) {
	tests := []struct {
		name   string
		format *core.ResponseFormat
		want   string
	}{
		{
			name:   "json object",
			format: &core.ResponseFormat{Type: "json_object"},
			want:   `{"type":"json_object"}`,
		},
		{
			name: "json schema without optional fields",
			format: &core.ResponseFormat{
				Type:       "json_schema",
				JSONSchema: &core.JSONSchemaFormat{Name: "answer", Schema: map[string]any{"type": "object"}},
			},
			want: `{"json_schema":{"name":"answer","schema":{"type":"object"}},"type":"json_schema"}`,
		},
		{
			name:   "json schema with no schema falls back to json object",
			format: &core.ResponseFormat{Type: "json_schema"},
			want:   `{"type":"json_object"}`,
		},
		{
			name:   "anything else is plain text",
			format: &core.ResponseFormat{Type: "text"},
			want:   `{"type":"text"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(convertResponseFormat(tt.format))
			if err != nil {
				t.Fatalf("marshaling response format: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("convertResponseFormat() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestConvertToolsDefaultsParameters(t *testing.T) {
	tools := convertTools([]core.Tool{{Function: core.Function{Name: "noop"}}})
	if len(tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(tools))
	}
	if tools[0].Function.Parameters == nil {
		t.Error("nil tool parameters were not defaulted to an empty schema")
	}
}

func TestConvertMessageFallbacks(t *testing.T) {
	tests := []struct {
		name string
		msg  core.Message
		want string
	}{
		{
			name: "tool message with no results still carries an id",
			msg:  core.Message{Role: core.RoleTool, Content: []core.Part{{Type: core.ContentText, Text: "orphan"}}},
			want: `{"content":"orphan","tool_call_id":"unknown","role":"tool"}`,
		},
		{
			name: "unknown role is sent as a user message",
			msg:  core.Message{Role: core.MessageRole("oracle"), Content: []core.Part{{Type: core.ContentText, Text: "hi"}}},
			want: `{"content":"hi","role":"user"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(convertMessage(tt.msg))
			if err != nil {
				t.Fatalf("marshaling message: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("convertMessage() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestConvertContentPartMissingPayload(t *testing.T) {
	tests := []struct {
		name string
		part core.Part
	}{
		{"image url part with no url", core.Part{Type: core.ContentImageURL}},
		{"image data part with no data", core.Part{Type: core.ContentImageData}},
		{"unhandled part type", core.Part{Type: core.ContentVideoURL, Text: "fallback"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(convertContentPart(tt.part))
			if err != nil {
				t.Fatalf("marshaling content part: %v", err)
			}
			// A part with nothing usable degrades to its text, never to a
			// malformed image reference.
			want := `{"text":"` + tt.part.Text + `","type":"text"}`
			if string(got) != want {
				t.Errorf("convertContentPart() = %s, want %s", got, want)
			}
		})
	}
}
