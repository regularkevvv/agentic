package codemode

import (
	"encoding/json"
	"errors"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
)

const wireVersion = 1

type wireResult struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Content any    `json:"content,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
	Error   string `json:"error,omitempty"`
}

type wireDisposition struct {
	Kind     agentic.ToolDispositionKind `json:"kind"`
	Result   *wireResult                 `json:"result,omitempty"`
	Continue bool                        `json:"continue,omitempty"`
}

type deferredProgram struct {
	Version      int                  `json:"version"`
	OuterCallID  string               `json:"outer_call_id"`
	Step         int                  `json:"step"`
	Checkpoint   Checkpoint           `json:"checkpoint"`
	Calls        []Call               `json:"calls"`
	Dispositions []wireDisposition    `json:"dispositions"`
	GateDeferral agentic.ToolDeferral `json:"gate_deferral"`
	Stdout       string               `json:"stdout,omitempty"`
}

type resumeResolution struct {
	CallID       string                   `json:"call_id"`
	Action       agentic.ToolResumeAction `json:"action"`
	OverrideArgs map[string]any           `json:"override_args,omitempty"`
	Result       *wireResult              `json:"result,omitempty"`
}

type resumePayload struct {
	Version     int                `json:"version"`
	OuterCallID string             `json:"outer_call_id"`
	Resolutions []resumeResolution `json:"resolutions"`
}

type operationPayload struct {
	Version int              `json:"version"`
	Step    int              `json:"step"`
	Call    *Call            `json:"call,omitempty"`
	Program *deferredProgram `json:"program,omitempty"`
	Result  *wireResult      `json:"result,omitempty"`
}

func cloneMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	var result map[string]any
	if json.Unmarshal(encoded, &result) != nil {
		return nil
	}
	return result
}

func cloneCalls(calls []Call) []Call {
	result := make([]Call, len(calls))
	for index, call := range calls {
		result[index] = call
		result[index].Input = cloneMap(call.Input)
	}
	return result
}

func encodeBounded(value any, maximum int, label string) ([]byte, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("encode %s: %w", label, err)
	}
	if len(encoded) > maximum {
		return nil, fmt.Errorf("%s exceeds %d bytes", label, maximum)
	}
	return encoded, nil
}

func cloneJSON(value any, maximum int, label string) (any, error) {
	if value == nil {
		return nil, nil
	}
	encoded, err := encodeBounded(value, maximum, label)
	if err != nil {
		return nil, err
	}
	var result any
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, fmt.Errorf("decode cloned %s: %w", label, err)
	}
	return result, nil
}

func toWireResult(result agentic.ToolExecutionResult, maximum int) (*wireResult, error) {
	content, err := cloneJSON(result.Content, maximum, "nested tool result")
	if err != nil {
		return nil, err
	}
	wire := &wireResult{
		ID:      result.ToolUseID,
		Name:    result.ToolName,
		Content: content,
		IsError: result.IsError,
	}
	if result.Error != nil {
		wire.Error = result.Error.Error()
	}
	return wire, nil
}

func fromWireResult(result *wireResult) (agentic.ToolExecutionResult, error) {
	if result == nil || result.ID == "" || result.Name == "" {
		return agentic.ToolExecutionResult{}, errors.New("invalid persisted nested tool result")
	}
	converted := agentic.ToolExecutionResult{
		ToolUseID: result.ID,
		ToolName:  result.Name,
		Content:   result.Content,
		IsError:   result.IsError,
	}
	if result.Error != "" {
		converted.Error = errors.New(result.Error)
	}
	return converted, nil
}
