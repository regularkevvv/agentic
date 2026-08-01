package agentic

import (
	"context"
	"encoding/json"
	"testing"
)

func TestWithToolCallContextStartsFreshNestedInvocation(t *testing.T) {
	outerCall := ToolCallContext{ID: "outer-1", Name: "run_code", Attempt: 1}
	outer := WithToolCallContext(context.Background(), outerCall)
	outer = withToolResumeContext(outer, ToolResumeContext{
		SuspensionID: "suspension-1",
		Deferral: ToolDeferral{
			Kind:    "nested.v1",
			Payload: json.RawMessage(`{"checkpoint":1}`),
		},
		Payload: json.RawMessage(`{"resolution":1}`),
	})

	nestedCall := ToolCallContext{ID: "nested-1", Name: "write_file", Attempt: 1}
	nested := WithToolCallContext(outer, nestedCall)
	if got, ok := CurrentToolCall(nested); !ok || got != nestedCall {
		t.Fatalf("nested call context = %#v, %t", got, ok)
	}
	if resume, ok := CurrentToolResume(nested); ok {
		t.Fatalf("nested invocation inherited outer resume context: %#v", resume)
	}

	if got, ok := CurrentToolCall(outer); !ok || got != outerCall {
		t.Fatalf("outer call context was mutated = %#v, %t", got, ok)
	}
	if resume, ok := CurrentToolResume(outer); !ok || resume.SuspensionID != "suspension-1" {
		t.Fatalf("outer resume context was mutated = %#v, %t", resume, ok)
	}
}
