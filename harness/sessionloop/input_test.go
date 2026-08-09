package sessionloop_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func TestValidateInputEnforcesPerKindBlockRules(t *testing.T) {
	t.Parallel()

	valid := []sessionloop.Input{
		{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}},
		{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockData, MediaType: "application/json", Data: json.RawMessage(`{"a":1}`)}}},
		{Blocks: []sessionloop.InputBlock{
			{Kind: sessionloop.InputBlockText, Text: "use this data"},
			{Kind: sessionloop.InputBlockData, MediaType: "application/json", Data: json.RawMessage(`{"a":1}`)},
		}},
	}
	for index, input := range valid {
		if err := sessionloop.ValidateInput(input); err != nil {
			t.Errorf("valid input %d rejected: %v", index, err)
		}
	}

	invalid := []struct {
		name  string
		input sessionloop.Input
	}{
		{"no blocks", sessionloop.Input{}},
		{"text without text", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText}}}},
		{"text with structured data", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "x", Data: json.RawMessage(`1`)}}}},
		{"text with media type", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "x", MediaType: "text/plain"}}}},
		{"data without data", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockData}}}},
		{"data with text", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockData, Text: "x", Data: json.RawMessage(`1`)}}}},
		{"truncated JSON data", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockData, Data: json.RawMessage(`{"a"`)}}}},
		{"two JSON values in one block", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockData, Data: json.RawMessage(`{} {}`)}}}},
		{"unknown block kind", sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: "hologram"}}}},
	}
	for _, violation := range invalid {
		if err := sessionloop.ValidateInput(violation.input); !errors.Is(err, sessionloop.ErrInvalidCommand) {
			t.Errorf("%s: ValidateInput = %v, want ErrInvalidCommand", violation.name, err)
		}
	}
}

func TestInputCloneIsDeeplyIndependent(t *testing.T) {
	t.Parallel()
	input := sessionloop.Input{
		Blocks: []sessionloop.InputBlock{
			{Kind: sessionloop.InputBlockText, Text: "keep"},
			{Kind: sessionloop.InputBlockData, Data: json.RawMessage(`{"a":1}`)},
		},
		Meta: map[string]string{"k": "v"},
	}
	clone := input.Clone()
	clone.Blocks[0].Text = "mutated"
	clone.Blocks[1].Data[2] = 'X'
	clone.Meta["k"] = "mutated"
	if input.Blocks[0].Text != "keep" || string(input.Blocks[1].Data) != `{"a":1}` || input.Meta["k"] != "v" {
		t.Fatalf("mutating the cloned input leaked into the original: %#v", input)
	}

	var empty sessionloop.Input
	emptyClone := empty.Clone()
	if emptyClone.Blocks != nil || emptyClone.Meta != nil {
		t.Fatalf("cloning an empty input invented content: %#v", emptyClone)
	}
}
