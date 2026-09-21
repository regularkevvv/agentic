// The deterministic model makes steering boundaries reproducible and records
// exact serialized request messages; it never substitutes for harness behavior.
package sessionloop_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

type modelRequest struct {
	messages []json.RawMessage
	tools    []byte
	cacheKey string
}

type scriptedModel struct {
	mu           sync.Mutex
	requests     []modelRequest
	entered      chan struct{}
	resume       chan struct{}
	readFile     bool
	toolVerified bool
}

func (*scriptedModel) Name() string { return "e2e:local-sessionloop" }

func (m *scriptedModel) Request(ctx context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	encoded, err := json.Marshal(request.Messages)
	if err != nil {
		return nil, err
	}
	var recorded modelRequest
	if err := json.Unmarshal(encoded, &recorded.messages); err != nil {
		return nil, err
	}
	recorded.tools, err = json.Marshal(request.Tools)
	if err != nil {
		return nil, err
	}
	if request.PromptCache != nil {
		recorded.cacheKey = request.PromptCache.Key
	}
	m.mu.Lock()
	m.requests = append(m.requests, recorded)
	call := len(m.requests)
	m.mu.Unlock()
	if call == 1 {
		close(m.entered)
		if m.resume != nil {
			select {
			case <-m.resume:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		if m.readFile {
			return &agentic.ChatResponse{Model: m.Name(), FinishReason: agentic.FinishReasonToolCalls,
				Message: agentic.NewToolUseMessage(agentic.ToolUse{ID: "read-fixture", Name: "read_file",
					Input: map[string]any{"path": "fixture.txt"}})}, nil
		}
	}
	if m.readFile && call == 2 {
		found := false
		for _, message := range request.Messages {
			for _, result := range message.GetToolResults() {
				if result.ToolUseID == "read-fixture" && !result.IsError && strings.Contains(result.Content, "local-e2e-content") {
					found = true
				}
			}
		}
		if !found || !bytes.Contains(encoded, []byte("include the fixture contents")) {
			return nil, errors.New("next model request did not contain the real tool result and steering")
		}
		m.mu.Lock()
		m.toolVerified = true
		m.mu.Unlock()
	}
	return &agentic.ChatResponse{Model: m.Name(), FinishReason: agentic.FinishReasonStop,
		Message: agentic.NewTextMessage(agentic.RoleAssistant, fmt.Sprintf("completed request %d", call))}, nil
}

func (m *scriptedModel) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

func (m *scriptedModel) assertContinuity(t *testing.T, sessionID string, count int) {
	t.Helper()
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.requests) != count {
		t.Fatalf("model requests=%d, want %d", len(m.requests), count)
	}
	if m.readFile && !m.toolVerified {
		t.Fatal("the model never received the real file contents together with steering")
	}
	for i, request := range m.requests {
		if request.cacheKey != sessionID {
			t.Fatalf("request %d changed prompt cache identity: %q", i, request.cacheKey)
		}
		if i == 0 {
			continue
		}
		previous := m.requests[i-1]
		if !bytes.Equal(previous.tools, request.tools) || len(request.messages) < len(previous.messages) {
			t.Fatalf("request %d changed tools or shortened history", i)
		}
		for j := range previous.messages {
			if !bytes.Equal(previous.messages[j], request.messages[j]) {
				t.Fatalf("request %d rewrote prefix message %d", i, j)
			}
		}
	}
}
