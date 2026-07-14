package bedrock

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestNewNoRegion(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_REGION", "")

	_, err := New("anthropic.claude-sonnet-4-20250514-v1:0")
	if err == nil {
		t.Error("expected error when no region is set")
	}
}

func TestNewWithRegion(t *testing.T) {
	model, err := New("anthropic.claude-sonnet-4-20250514-v1:0", WithRegion("us-east-1"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "anthropic.claude-sonnet-4-20250514-v1:0" {
		t.Errorf("expected name %q, got %q", "anthropic.claude-sonnet-4-20250514-v1:0", model.Name())
	}
}

func TestNewWithClient(t *testing.T) {
	// Using WithClient bypasses AWS config loading entirely.
	client := bedrockruntime.New(bedrockruntime.Options{Region: "us-east-1"})
	model, err := New("meta.llama3-1-70b-instruct-v1:0", WithClient(client))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "meta.llama3-1-70b-instruct-v1:0" {
		t.Errorf("expected name %q, got %q", "meta.llama3-1-70b-instruct-v1:0", model.Name())
	}
}

func TestNewWithCredentials(t *testing.T) {
	model, err := New("anthropic.claude-sonnet-4-20250514-v1:0",
		WithRegion("us-east-1"),
		WithCredentials("AKID", "SECRET", "SESSION"),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "anthropic.claude-sonnet-4-20250514-v1:0" {
		t.Errorf("expected model name, got %q", model.Name())
	}
}

func TestNewFromEnvRegion(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "eu-west-1")

	model, err := New("anthropic.claude-sonnet-4-20250514-v1:0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.Name() != "anthropic.claude-sonnet-4-20250514-v1:0" {
		t.Errorf("expected model name, got %q", model.Name())
	}
}

func TestMustNew(t *testing.T) {
	model := MustNew("anthropic.claude-sonnet-4-20250514-v1:0", WithRegion("us-east-1"))
	if model.Name() != "anthropic.claude-sonnet-4-20250514-v1:0" {
		t.Errorf("expected model name, got %q", model.Name())
	}
}

func TestMustNewPanics(t *testing.T) {
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_REGION", "")

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when no region is set")
		}
	}()
	MustNew("anthropic.claude-sonnet-4-20250514-v1:0")
}

func TestConvertRole(t *testing.T) {
	tests := []struct {
		role     core.MessageRole
		expected types.ConversationRole
	}{
		{core.RoleAssistant, types.ConversationRoleAssistant},
		{core.RoleUser, types.ConversationRoleUser},
		{core.RoleTool, types.ConversationRoleUser},
	}

	for _, tt := range tests {
		got := convertRole(tt.role)
		if got != tt.expected {
			t.Errorf("convertRole(%q) = %q, want %q", tt.role, got, tt.expected)
		}
	}
}

func TestConvertStopReason(t *testing.T) {
	tests := []struct {
		reason   types.StopReason
		expected core.FinishReason
	}{
		{types.StopReasonEndTurn, core.FinishReasonStop},
		{types.StopReasonStopSequence, core.FinishReasonStop},
		{types.StopReasonMaxTokens, core.FinishReasonLength},
		{types.StopReasonToolUse, core.FinishReasonToolCalls},
		{types.StopReasonContentFiltered, core.FinishReasonContentFilter},
		{types.StopReasonGuardrailIntervened, core.FinishReasonContentFilter},
	}

	for _, tt := range tests {
		got := convertStopReason(tt.reason)
		if got != tt.expected {
			t.Errorf("convertStopReason(%q) = %q, want %q", tt.reason, got, tt.expected)
		}
	}
}

func TestImageFormat(t *testing.T) {
	tests := []struct {
		mediaType string
		expected  types.ImageFormat
	}{
		{"image/png", types.ImageFormatPng},
		{"image/gif", types.ImageFormatGif},
		{"image/webp", types.ImageFormatWebp},
		{"image/jpeg", types.ImageFormatJpeg},
		{"image/unknown", types.ImageFormatJpeg},
	}

	for _, tt := range tests {
		got := imageFormat(tt.mediaType)
		if got != tt.expected {
			t.Errorf("imageFormat(%q) = %q, want %q", tt.mediaType, got, tt.expected)
		}
	}
}

func TestImplementsInterfaces(t *testing.T) {
	model, err := New("anthropic.claude-sonnet-4-20250514-v1:0", WithRegion("us-east-1"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var _ core.Model = model
	var _ core.StreamModel = model
}
