// Command main is the fresh no-replace consumer proof for the realtime
// release view: it bridges a scripted call to the sessionloop reference host,
// answers one delegate call with the session's committed reply, and exits 0.
// Its module graph must contain nothing beyond realtime and the sessionloop
// protocol it declares — no Agentic, Harness, provider SDK, or WebRTC stack.
package main

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
	"github.com/regularkevvv/agentic/realtime"
)

// call is a scripted voice call: events in, the bridge's actions out.
type call struct {
	events  chan realtime.Event
	actions chan realtime.Action
}

func (c call) Send(_ context.Context, action realtime.Action) error {
	c.actions <- action
	return nil
}

func (c call) Recv(ctx context.Context) (realtime.Event, error) {
	select {
	case event, ok := <-c.events:
		if !ok {
			return realtime.Event{}, io.EOF
		}
		return event, nil
	case <-ctx.Done():
		return realtime.Event{}, ctx.Err()
	}
}

func (call) Close() error { return nil }

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	session, err := testkit.New().NewSession(ctx, sessionloop.SessionOptions{})
	if err != nil {
		panic(err)
	}
	defer func() { _ = session.Close(context.Background()) }()

	c := call{events: make(chan realtime.Event, 4), actions: make(chan realtime.Action, 8)}
	done := make(chan error, 1)
	go func() { done <- realtime.Run(ctx, c, session, realtime.Options{}) }()

	if configure := <-c.actions; configure.Kind != realtime.ActionConfigure || len(configure.Config.Tools) != 1 {
		panic(fmt.Sprintf("incompatible configuration: %+v", configure))
	}
	c.events <- realtime.Event{Kind: realtime.EventToolCall, Call: &realtime.ToolCall{
		ID: "call-1", Name: realtime.DelegateTool, Arguments: []byte(`{"request":"hello"}`),
	}}
	result := <-c.actions
	if result.Kind != realtime.ActionToolResult || result.Result.CallID != "call-1" || result.Result.Output != "echo: hello" {
		panic(fmt.Sprintf("incompatible tool result: %+v", result))
	}
	if respond := <-c.actions; respond.Kind != realtime.ActionRespond {
		panic(fmt.Sprintf("incompatible follow-up: %+v", respond))
	}
	close(c.events)
	if err := <-done; err != io.EOF {
		panic(fmt.Sprintf("bridge ended with %v, want io.EOF", err))
	}

	snapshot, err := session.Snapshot(ctx)
	if err != nil || len(snapshot.Entries) != 2 {
		panic(fmt.Sprintf("session did not own the exchange: %+v %v", snapshot.Entries, err))
	}
	fmt.Printf("compatible_realtime session=%s reply=%q\n", snapshot.SessionID, result.Result.Output)
}
