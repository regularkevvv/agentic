package testutil

import (
	"context"
	"errors"

	"github.com/regularkevvv/agentic/internal/core"
)

// StubModel is a small request spy for tests that only need a canned response
// or error and want to inspect the requests that were made.
type StubModel struct {
	NameValue string
	Response  *core.ChatResponse
	Err       error
	Requests  []*core.ChatRequest
}

func (m *StubModel) Request(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	m.Requests = append(m.Requests, req)
	if m.Err != nil {
		return nil, m.Err
	}
	if m.Response != nil {
		return m.Response, nil
	}
	return &core.ChatResponse{Model: m.Name()}, nil
}

func (m *StubModel) Name() string {
	if m.NameValue != "" {
		return m.NameValue
	}
	return "stub-model"
}

// ScriptedStreamModel is a request spy for streaming tests that replays a fixed
// sequence of stream event slices across successive RequestStream calls.
type ScriptedStreamModel struct {
	NameValue string
	Requests  []*core.ChatRequest
	Streams   [][]core.StreamEvent
}

func (m *ScriptedStreamModel) Request(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	return nil, errors.New("Request should not be called for streaming tests")
}

func (m *ScriptedStreamModel) RequestStream(ctx context.Context, req *core.ChatRequest) (*core.StreamResult, error) {
	m.Requests = append(m.Requests, req)
	if len(m.Streams) == 0 {
		return nil, errors.New("no scripted stream available")
	}

	events := m.Streams[0]
	m.Streams = m.Streams[1:]
	return NewScriptedStream(events...), nil
}

func (m *ScriptedStreamModel) Name() string {
	if m.NameValue != "" {
		return m.NameValue
	}
	return "scripted-stream"
}

func NewScriptedStream(events ...core.StreamEvent) *core.StreamResult {
	ch := make(chan core.StreamEvent, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return core.NewStreamResult(ch)
}
