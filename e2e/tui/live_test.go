//go:build e2e

package tui_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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
	systemPrompt := `You are running a live Harness/TUI acceptance test. When the operator asks for a shell command, call run_command exactly once with name /bin/sh and the exact -c argument supplied. Never simulate execution. After the tool result, answer briefly.` + "\n" +
		strings.Repeat("Stable cache anchor: preserve this instruction prefix exactly across every turn.\n", 768)
	if err := os.WriteFile(prompt, []byte(systemPrompt), 0o600); err != nil {
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
	markerPath := filepath.Join(workspace, marker)
	if err := session.Submit(ctx, uit.Input{Text: "Call run_command with name /bin/sh and args [-c, 'printf live-denied > " + marker + "']. I will deny permission."}); err != nil {
		t.Fatal(err)
	}
	denied, err := session.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if denied.State != uit.StateSuspended || denied.Suspension == nil || len(denied.Suspension.Approvals) != 1 {
		t.Fatalf("provider did not reach the denial suspension: %#v", denied)
	}
	approval := denied.Suspension.Approvals[0]
	if err := session.Resolve(ctx, uit.Resolution{
		SuspensionID: denied.Suspension.ID,
		Decisions:    []uit.Decision{{CallID: approval.CallID, Action: uit.DecisionDeny, Reason: "live e2e denial"}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(markerPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("denied command changed marker state: %v", err)
	}

	if err := session.Submit(ctx, uit.Input{Text: "Call run_command with name /bin/sh and args [-c, 'printf live-approved > " + marker + "']. I will approve permission."}); err != nil {
		t.Fatal(err)
	}
	approved, err := session.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if approved.State != uit.StateSuspended || approved.Suspension == nil || len(approved.Suspension.Approvals) != 1 {
		t.Fatalf("provider did not reach the approval suspension: %#v", approved)
	}
	approval = approved.Suspension.Approvals[0]
	if err := session.Resolve(ctx, uit.Resolution{
		SuspensionID: approved.Suspension.ID,
		Decisions:    []uit.Decision{{CallID: approval.CallID, Action: uit.DecisionApprove}},
	}); err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(markerPath)
	if err != nil || string(contents) != "live-approved" {
		t.Fatalf("live marker = %q, %v", contents, err)
	}
	beforeClose, err := session.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if beforeClose.State != uit.StateIdle || beforeClose.Usage.Requests < 4 {
		t.Fatalf("live flow did not finish four provider requests: %#v", beforeClose)
	}
	if beforeClose.Usage.CacheReadTokens == 0 || beforeClose.Usage.CacheHitPercent() < 25 {
		t.Fatalf("live prompt cache was ineffective: usage=%#v hit=%.1f%%", beforeClose.Usage, beforeClose.Usage.CacheHitPercent())
	}
	t.Logf("live usage: requests=%d prompt=%d cache_read=%d cache_created=%d hit=%.1f%%",
		beforeClose.Usage.Requests,
		beforeClose.Usage.PromptTokens,
		beforeClose.Usage.CacheReadTokens,
		beforeClose.Usage.CacheCreationTokens,
		beforeClose.Usage.CacheHitPercent(),
	)

	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	for events.Load() < 8 {
		select {
		case <-deadline.C:
			t.Fatalf("live flow emitted only %d TUI events", events.Load())
		case <-time.After(10 * time.Millisecond):
		}
	}
	subscription.Close()
	<-eventDone
	if value := observationErr.Load(); value != nil {
		t.Fatal(value.(error))
	}
	if err := session.Close(ctx); err != nil {
		t.Fatal(err)
	}
	reopened, err := assembly.Host.ResumeSession(ctx, session.ID())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = reopened.Close(context.WithoutCancel(ctx)) }()
	afterReopen, err := reopened.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if afterReopen.Cursor < beforeClose.Cursor || !reflect.DeepEqual(afterReopen.Transcript, beforeClose.Transcript) || afterReopen.Usage != beforeClose.Usage {
		t.Fatalf("live durable reopen mismatch: before=%#v after=%#v", beforeClose, afterReopen)
	}
}
