//go:build e2e

package tui_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	uit "github.com/regularkevvv/agentic/tui"
	"github.com/regularkevvv/agentic/tui/standard"
)

func TestLiveProviderHarnessTUIFlow(t *testing.T) {
	if os.Getenv("AGENTIC_TUI_LIVE") != "1" {
		t.Skip("set AGENTIC_TUI_LIVE=1 with provider credentials to run")
	}
	provider := os.Getenv("AGENTIC_TUI_LIVE_PROVIDER")
	model := os.Getenv("AGENTIC_TUI_LIVE_MODEL")
	if provider == "" || model == "" {
		t.Fatal("AGENTIC_TUI_LIVE_PROVIDER and AGENTIC_TUI_LIVE_MODEL are required")
	}
	workspace := t.TempDir()
	prompt := filepath.Join(t.TempDir(), "system.md")
	if err := os.WriteFile(prompt, []byte("Follow the operator's request. Use run_command when explicitly asked."), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()
	registry, err := standard.NewRegistry(standard.BuiltinFactories(nil)...)
	if err != nil {
		t.Fatal(err)
	}
	assembly, err := standard.Build(ctx, registry, standard.Config{
		ProfileName: "live-smoke", Provider: provider, Model: model,
		ContextWindowTokens: 128_000, SystemPromptFile: prompt,
		WorkspaceRoot: workspace, SessionDirectory: filepath.Join(t.TempDir(), "sessions"),
		Permission: "workspace-write", PromptCacheRetention: agentic.PromptCacheShort,
	})
	if err != nil {
		t.Fatal(err)
	}
	session, err := assembly.Host.NewSession(ctx, uit.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close(context.WithoutCancel(ctx)) }()
	subscription := session.Subscribe(uit.SubscribeOptions{Buffer: 512, Preview: true})
	defer subscription.Close()
	var events atomic.Int32
	eventDone := make(chan struct{})
	var observationErr atomic.Value
	go func() {
		defer close(eventDone)
		for eventStream, errorsStream := subscription.Events(), subscription.Errors(); eventStream != nil || errorsStream != nil; {
			select {
			case _, ok := <-eventStream:
				if !ok {
					eventStream = nil
				} else {
					events.Add(1)
				}
			case value, ok := <-errorsStream:
				if !ok {
					errorsStream = nil
				} else if value != nil {
					observationErr.Store(value)
				}
			}
		}
	}()

	marker := "tui-live-marker.txt"
	request := "Run /bin/sh with arguments -c and 'printf live-approved > " + marker + "'. Do not simulate it; request the run_command tool."
	if err := session.Submit(ctx, uit.Input{Text: request}); err != nil {
		t.Fatal(err)
	}
	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.State != uit.StateSuspended || snapshot.Suspension == nil || len(snapshot.Suspension.Approvals) != 1 {
		t.Fatalf("provider did not reach one permission suspension: %#v", snapshot)
	}
	approval := snapshot.Suspension.Approvals[0]
	if err := session.Resolve(ctx, uit.Resolution{
		SuspensionID: snapshot.Suspension.ID,
		Decisions:    []uit.Decision{{CallID: approval.CallID, Action: uit.DecisionApprove}},
	}); err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(filepath.Join(workspace, marker))
	if err != nil || string(contents) != "live-approved" {
		t.Fatalf("live marker = %q, %v", contents, err)
	}
	if value := observationErr.Load(); value != nil {
		t.Fatal(value.(error))
	}
	if events.Load() == 0 {
		t.Fatal(errors.New("live flow emitted no TUI events"))
	}
}
