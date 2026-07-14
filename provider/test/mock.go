package test

import (
	"context"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
)

// ModelResponse is a pre-configured response for the TestModel.
type ModelResponse struct {
	Text      string
	ToolCalls []core.ToolUse
	Usage     *core.Usage // Optional: if nil, default usage (10/5/15) is used.
}

// TestModel is a mock Model implementation for testing agents without API calls.
type TestModel struct {
	name      string
	responses []ModelResponse
	calls     []core.ChatRequest
	callIndex int
}

// NewTestModel creates a TestModel that returns the given responses in order.
// When responses are exhausted, it returns the last one repeatedly.
func NewTestModel(responses ...ModelResponse) *TestModel {
	if len(responses) == 0 {
		responses = []ModelResponse{{Text: "test response"}}
	}
	return &TestModel{
		name:      "test:mock",
		responses: responses,
	}
}

// Request implements Model.
func (m *TestModel) Request(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	m.calls = append(m.calls, *req)

	idx := m.callIndex
	if idx >= len(m.responses) {
		idx = len(m.responses) - 1
	}
	m.callIndex++

	resp := m.responses[idx]

	// Build message parts
	var parts []core.Part
	if resp.Text != "" {
		parts = append(parts, core.Part{
			Type: core.ContentText,
			Text: resp.Text,
		})
	}
	for _, tc := range resp.ToolCalls {
		tc := tc
		if tc.ID == "" {
			tc.ID = fmt.Sprintf("call_%d", m.callIndex)
		}
		parts = append(parts, core.Part{
			Type:    core.ContentToolUse,
			ToolUse: &tc,
		})
	}

	finishReason := core.FinishReasonStop
	if len(resp.ToolCalls) > 0 {
		finishReason = core.FinishReasonToolCalls
	}

	return &core.ChatResponse{
		ID:    fmt.Sprintf("test-%d", m.callIndex),
		Model: m.name,
		Choices: []core.Choice{
			{
				Index: 0,
				Message: core.Message{
					Role:    core.RoleAssistant,
					Content: parts,
				},
				FinishReason: finishReason,
			},
		},
		Usage: func() core.Usage {
			if resp.Usage != nil {
				return *resp.Usage
			}
			return core.Usage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			}
		}(),
		Created: time.Now(),
	}, nil
}

// Name implements Model.
func (m *TestModel) Name() string {
	return m.name
}

// Calls returns all requests made to this model.
func (m *TestModel) Calls() []core.ChatRequest {
	return m.calls
}

// CallCount returns the number of requests made.
func (m *TestModel) CallCount() int {
	return len(m.calls)
}

// Reset clears the call history and resets the response index.
func (m *TestModel) Reset() {
	m.calls = nil
	m.callIndex = 0
}
