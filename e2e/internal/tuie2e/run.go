// Package tuie2e contains the credential-free scenario shared by the TUI
// example and its end-to-end acceptance test.
package tuie2e

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/permission"
	uit "github.com/regularkevvv/agentic/tui"
	tuiharness "github.com/regularkevvv/agentic/tui/adapter/harness"
)

const (
	DeniedOutput   = "The first command was denied without running."
	ApprovedOutput = "The approved command ran and wrote the marker."
	markerName     = "approval-marker.txt"
)

// Report exposes proof from the model, runtime, permission, observation,
// durable-recovery, and provider-cache boundaries crossed by Run.
type Report struct {
	SessionID          string
	Cursor             uint64
	TranscriptEntries  int
	TextPreviews       int
	ToolEvents         int
	SuspensionEvents   int
	PromptTokens       int
	CacheReadTokens    int
	CacheHitPercent    float64
	ModelRequests      int
	StableCacheKeys    bool
	AppendOnlyPrefixes bool
	DeniedMarkerAbsent bool
	ApprovedMarker     string
}

// Run drives a real local Harness through the public TUI adapter. The model is
// deterministic, but every other boundary is production code.
func Run(ctx context.Context, workspace, sessionDir string) (report Report, err error) {
	if workspace == "" || sessionDir == "" {
		return Report{}, errors.New("workspace and session directory are required")
	}
	policy, err := permission.New(permission.DecisionDeny,
		permission.Rule{Pattern: "shell/**", Decision: permission.DecisionAsk},
	)
	if err != nil {
		return Report{}, fmt.Errorf("construct permission policy: %w", err)
	}
	model := &scriptedStreamModel{}
	agent := agentic.NewAgent(
		"Use run_command exactly as requested and report whether it completed.",
		model,
	)
	runtime, err := harness.Default(agent, harness.DefaultConfig{
		WorkspaceRoot:        workspace,
		SessionDir:           sessionDir,
		ContextWindowTokens:  16_384,
		PromptCacheRetention: agentic.PromptCacheShort,
		ModelStreaming:       true,
		PermissionPolicy:     policy,
	})
	if err != nil {
		return Report{}, fmt.Errorf("construct Harness: %w", err)
	}
	host, err := tuiharness.New(runtime, tuiharness.WithProfileLabel("e2e:scripted"), tuiharness.WithWorkspace(workspace))
	if err != nil {
		return Report{}, fmt.Errorf("construct TUI adapter: %w", err)
	}
	session, err := host.NewSession(ctx, uit.SessionOptions{})
	if err != nil {
		return Report{}, fmt.Errorf("create TUI session: %w", err)
	}
	report.SessionID = session.ID()
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, session.Close(context.WithoutCancel(ctx)))
		}
	}()

	collector := newEventCollector(session.Subscribe(uit.SubscribeOptions{Buffer: 512, Preview: true}))
	defer collector.close()

	marker := filepath.Join(workspace, markerName)
	if removeErr := os.Remove(marker); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
		return report, fmt.Errorf("remove stale marker: %w", removeErr)
	}
	if err := session.Submit(ctx, uit.Input{Text: "Try the command, but I will deny permission."}); err != nil {
		return report, fmt.Errorf("submit denied turn: %w", err)
	}
	denied, err := session.Snapshot(ctx)
	if err != nil {
		return report, fmt.Errorf("snapshot denied suspension: %w", err)
	}
	if denied.State != uit.StateSuspended || denied.Suspension == nil || !denied.Suspension.Supported || len(denied.Suspension.Approvals) != 1 {
		return report, fmt.Errorf("denied turn did not expose one supported approval: %#v", denied)
	}
	approval := denied.Suspension.Approvals[0]
	if err := session.Resolve(ctx, uit.Resolution{
		SuspensionID: denied.Suspension.ID,
		Decisions:    []uit.Decision{{CallID: approval.CallID, Action: uit.DecisionDeny, Reason: "e2e denial"}},
	}); err != nil {
		return report, fmt.Errorf("deny command: %w", err)
	}
	_, statErr := os.Stat(marker)
	report.DeniedMarkerAbsent = errors.Is(statErr, os.ErrNotExist)
	if !report.DeniedMarkerAbsent {
		return report, fmt.Errorf("denied command changed marker state: %v", statErr)
	}

	if err := session.Submit(ctx, uit.Input{Text: "Try again; I will approve this command."}); err != nil {
		return report, fmt.Errorf("submit approved turn: %w", err)
	}
	approved, err := session.Snapshot(ctx)
	if err != nil {
		return report, fmt.Errorf("snapshot approved suspension: %w", err)
	}
	if approved.State != uit.StateSuspended || approved.Suspension == nil || len(approved.Suspension.Approvals) != 1 {
		return report, fmt.Errorf("approved turn did not expose one approval: %#v", approved)
	}
	approval = approved.Suspension.Approvals[0]
	if err := session.Resolve(ctx, uit.Resolution{
		SuspensionID: approved.Suspension.ID,
		Decisions:    []uit.Decision{{CallID: approval.CallID, Action: uit.DecisionApprove}},
	}); err != nil {
		return report, fmt.Errorf("approve command: %w", err)
	}
	contents, err := os.ReadFile(marker)
	if err != nil {
		return report, fmt.Errorf("read approved marker: %w", err)
	}
	report.ApprovedMarker = string(contents)
	if report.ApprovedMarker != "approved" {
		return report, fmt.Errorf("approved marker = %q", report.ApprovedMarker)
	}

	beforeClose, err := session.Snapshot(ctx)
	if err != nil {
		return report, fmt.Errorf("snapshot completed session: %w", err)
	}
	if beforeClose.State != uit.StateIdle {
		return report, fmt.Errorf("completed session state = %s", beforeClose.State)
	}
	if err := collector.wait(ctx, 2, 2, 4); err != nil {
		return report, err
	}
	if err := session.Close(ctx); err != nil {
		return report, fmt.Errorf("close session: %w", err)
	}
	closed = true
	collector.close()

	reopened, err := host.ResumeSession(ctx, report.SessionID)
	if err != nil {
		return report, fmt.Errorf("reopen session: %w", err)
	}
	defer func() { err = errors.Join(err, reopened.Close(context.WithoutCancel(ctx))) }()
	afterReopen, err := reopened.Snapshot(ctx)
	if err != nil {
		return report, fmt.Errorf("snapshot reopened session: %w", err)
	}
	if afterReopen.Cursor < beforeClose.Cursor || !reflect.DeepEqual(afterReopen.Transcript, beforeClose.Transcript) {
		return report, errors.New("durable close/reopen regressed the cursor or changed the transcript")
	}

	report.Cursor = afterReopen.Cursor
	report.TranscriptEntries = len(afterReopen.Transcript)
	report.PromptTokens = afterReopen.Usage.PromptTokens
	report.CacheReadTokens = afterReopen.Usage.CacheReadTokens
	report.CacheHitPercent = afterReopen.Usage.CacheHitPercent()
	report.TextPreviews, report.ToolEvents, report.SuspensionEvents = collector.counts()
	report.ModelRequests, report.StableCacheKeys, report.AppendOnlyPrefixes = model.proof(report.SessionID)
	if report.TranscriptEntries < 8 || report.ModelRequests != 4 || !report.StableCacheKeys || !report.AppendOnlyPrefixes ||
		report.TextPreviews < 2 || report.ToolEvents < 2 || report.SuspensionEvents < 2 || report.CacheReadTokens == 0 {
		return report, fmt.Errorf("incomplete TUI e2e evidence: %#v", report)
	}
	return report, nil
}

type scriptedStreamModel struct {
	mu       sync.Mutex
	requests []*agentic.ChatRequest
}

func (*scriptedStreamModel) Name() string { return "e2e:tui-scripted-stream" }

func (*scriptedStreamModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return nil, errors.New("non-streaming request is not supported")
}

func (m *scriptedStreamModel) RequestStream(_ context.Context, request *agentic.ChatRequest) (*agentic.StreamResult, error) {
	m.mu.Lock()
	m.requests = append(m.requests, request)
	call := len(m.requests)
	m.mu.Unlock()
	usage := &agentic.Usage{PromptTokens: 100, CompletionTokens: 10, TotalTokens: 110, CacheReadTokens: 80, Requests: 1}
	var events []agentic.StreamEvent
	switch call {
	case 1:
		events = commandStream("denied-command", `{"name":"/bin/sh","args":["-c","printf denied > approval-marker.txt"]}`, usage)
	case 2:
		events = textStream("The first command ", "was denied without running.", usage)
	case 3:
		events = commandStream("approved-command", `{"name":"/bin/sh","args":["-c","printf approved > approval-marker.txt"]}`, usage)
	case 4:
		events = textStream("The approved command ran ", "and wrote the marker.", usage)
	default:
		return nil, fmt.Errorf("unexpected model request %d", call)
	}
	stream := make(chan agentic.StreamEvent, len(events))
	for _, event := range events {
		stream <- event
	}
	close(stream)
	return agentic.NewStreamResult(stream), nil
}

func commandStream(id, arguments string, usage *agentic.Usage) []agentic.StreamEvent {
	return []agentic.StreamEvent{
		{Type: agentic.StreamEventToolCallStart, ToolUse: &agentic.ToolUse{ID: id, Name: "run_command"}},
		{Type: agentic.StreamEventToolCallDelta, ToolCallID: id, Delta: arguments},
		{Type: agentic.StreamEventDone, FinishReason: agentic.FinishReasonToolCalls, Usage: usage},
	}
}

func textStream(first, second string, usage *agentic.Usage) []agentic.StreamEvent {
	return []agentic.StreamEvent{
		{Type: agentic.StreamEventTextDelta, Delta: first},
		{Type: agentic.StreamEventTextDelta, Delta: second},
		{Type: agentic.StreamEventDone, FinishReason: agentic.FinishReasonStop, Usage: usage},
	}
}

func (m *scriptedStreamModel) proof(sessionID string) (int, bool, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	stable := len(m.requests) > 0
	appendOnly := len(m.requests) > 0
	for index, request := range m.requests {
		stable = stable && request.PromptCache != nil && request.PromptCache.Key == sessionID &&
			request.PromptCache.Retention == agentic.PromptCacheShort
		if index > 0 {
			previous := m.requests[index-1].Messages
			appendOnly = appendOnly && len(request.Messages) >= len(previous) && reflect.DeepEqual(previous, request.Messages[:len(previous)])
		}
	}
	return len(m.requests), stable, appendOnly
}

type eventCollector struct {
	subscription uit.Subscription
	done         chan struct{}
	mu           sync.Mutex
	text         int
	tools        int
	suspensions  int
	err          error
	once         sync.Once
}

func newEventCollector(subscription uit.Subscription) *eventCollector {
	collector := &eventCollector{subscription: subscription, done: make(chan struct{})}
	go func() {
		defer close(collector.done)
		for events, errs := subscription.Events(), subscription.Errors(); events != nil || errs != nil; {
			select {
			case event, ok := <-events:
				if !ok {
					events = nil
					continue
				}
				collector.mu.Lock()
				switch event.Kind {
				case uit.EventTextDelta:
					collector.text++
				case uit.EventToolPlanned, uit.EventToolStarted, uit.EventToolResult:
					collector.tools++
				case uit.EventRunSuspended:
					collector.suspensions++
				}
				collector.mu.Unlock()
			case value, ok := <-errs:
				if !ok {
					errs = nil
					continue
				}
				if value != nil {
					collector.mu.Lock()
					collector.err = value
					collector.mu.Unlock()
				}
			}
		}
	}()
	return collector
}

func (c *eventCollector) counts() (int, int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.text, c.tools, c.suspensions
}

func (c *eventCollector) wait(ctx context.Context, suspensions, tools, text int) error {
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		observedText, observedTools, observedSuspensions := c.counts()
		c.mu.Lock()
		observedErr := c.err
		c.mu.Unlock()
		if observedErr != nil {
			return fmt.Errorf("observe TUI session: %w", observedErr)
		}
		if observedSuspensions >= suspensions && observedTools >= tools && observedText >= text {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("wait for TUI observations: %w", ctx.Err())
		case <-c.done:
			return fmt.Errorf("TUI observation ended at text=%d tools=%d suspensions=%d", observedText, observedTools, observedSuspensions)
		case <-ticker.C:
		}
	}
}

func (c *eventCollector) close() {
	c.once.Do(func() {
		c.subscription.Close()
		<-c.done
	})
}
