package app

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"

	uit "github.com/regularkevvv/agentic/tui"
	"github.com/regularkevvv/agentic/tui/internal/testhost"
	"github.com/regularkevvv/agentic/tui/render"
)

func key(code rune, text string, mod tea.KeyMod) tea.KeyPressMsg {
	return tea.KeyPressMsg(tea.Key{Code: code, Text: text, Mod: mod})
}

func readyModel(t *testing.T, script testhost.Script) *Model {
	t.Helper()
	model, err := New(testhost.New(script), Options{Config: Config{
		AlternateScreen: AlternateNever, NoColor: true, Thinking: render.ThinkingCollapsed,
		PreviewHz: 240, EventBuffer: 64,
	}, Environ: []string{"TERM=xterm"}})
	if err != nil {
		t.Fatal(err)
	}
	message := model.Init()()
	updated, _ := model.Update(message)
	result := updated.(*Model)
	t.Cleanup(func() {
		if result.bridge != nil {
			result.bridge.Close()
		}
		if result.session != nil {
			_ = result.session.Close(context.Background())
		}
		result.cancel()
	})
	return result
}

func execute(t *testing.T, model *Model, command tea.Cmd) {
	t.Helper()
	if command == nil {
		t.Fatal("expected command")
	}
	message := command()
	model.Update(message)
}

func TestNewConfigAndAltScreenValidation(t *testing.T) {
	t.Parallel()
	if _, err := New(nil, Options{}); err == nil {
		t.Fatal("nil host succeeded")
	}
	host := testhost.New(nil)
	invalid := []Config{
		{AlternateScreen: "sometimes", Thinking: render.ThinkingCollapsed},
		{AlternateScreen: AlternateNever, Thinking: "opaque"},
		{AlternateScreen: AlternateNever, Thinking: render.ThinkingHidden, PreviewHz: 241},
		{AlternateScreen: AlternateNever, Thinking: render.ThinkingHidden, EventBuffer: -1},
	}
	for index, config := range invalid {
		if _, err := New(host, Options{Config: config}); err == nil {
			t.Fatalf("invalid config %d succeeded", index)
		}
	}
	if !ResolveAltScreen(AlternateAlways, nil) || ResolveAltScreen(AlternateNever, nil) || ResolveAltScreen(AlternateAuto, []string{"TMUX=1", "TERM=xterm"}) || ResolveAltScreen(AlternateAuto, []string{"STY=1"}) || ResolveAltScreen(AlternateAuto, []string{"TERM=dumb"}) || !ResolveAltScreen(AlternateAuto, []string{"TERM=xterm"}) {
		t.Fatal("alternate screen resolution incorrect")
	}
	model, err := New(host, Options{Environ: []string{"NO_COLOR=1", "TERM=xterm"}})
	if err != nil || !model.config.NoColor || model.composer.VirtualCursor() {
		t.Fatalf("NO_COLOR model = %#v, %v", model, err)
	}
	model.cancel()
}

func TestSubmitPasteQueueHistoryAndNonblockingUpdate(t *testing.T) {
	t.Parallel()
	release := make(chan struct{})
	model := readyModel(t, func(ctx context.Context, session *testhost.Session, input uit.Input) error {
		<-release
		session.Complete("done", uit.Usage{PromptTokens: 10, CacheReadTokens: 5})
		return nil
	})
	model.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	model.composer.SetValue("first line")
	model.Update(key(tea.KeyEnter, "", tea.ModShift))
	model.Update(key('j', "j", tea.ModCtrl))
	if model.composer.Value() != "first line\n\n" {
		t.Fatalf("multiline composer = %q", model.composer.Value())
	}
	model.composer.Reset()
	largePaste := strings.Repeat("large paste line\n", 4096)
	model.Update(tea.PasteMsg{Content: largePaste})
	if model.composer.Value() != largePaste {
		t.Fatalf("large bracketed paste was truncated: got=%d want=%d", len(model.composer.Value()), len(largePaste))
	}
	start := time.Now()
	_, command := model.Update(key(tea.KeyEnter, "", 0))
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Fatalf("Update blocked for %v", elapsed)
	}
	if command == nil || model.composer.Value() != "" || len(model.history) != 1 || model.history[0] != largePaste {
		t.Fatalf("submit dispatch state = value %q history %#v", model.composer.Value(), model.history)
	}
	done := make(chan tea.Msg, 1)
	go func() { done <- command() }()
	model.composer.SetValue("steer now")
	_, steer := model.Update(key('s', "s", tea.ModAlt))
	execute(t, model, steer)
	model.composer.SetValue("follow")
	_, follow := model.Update(key('f', "f", tea.ModAlt))
	execute(t, model, follow)
	model.composer.SetValue("next")
	_, next := model.Update(key('n', "n", tea.ModAlt))
	execute(t, model, next)
	close(release)
	select {
	case message := <-done:
		model.Update(message)
	case <-time.After(time.Second):
		t.Fatal("submit operation did not complete")
	}
	if len(model.snapshot.Pending) != 3 || model.snapshot.State != uit.StateIdle {
		t.Fatalf("snapshot = %#v", model.snapshot)
	}
	model.composer.Reset()
	model.Update(key(tea.KeyUp, "", 0))
	if model.composer.Value() != "next" {
		t.Fatalf("history recall = %q", model.composer.Value())
	}
	model.Update(key(tea.KeyUp, "", 0))
	if model.composer.Value() != "follow" {
		t.Fatalf("history previous = %q", model.composer.Value())
	}
	model.Update(key(tea.KeyDown, "", 0))
	if model.composer.Value() != "next" {
		t.Fatalf("history next = %q", model.composer.Value())
	}
	model.Update(key(tea.KeyDown, "", 0))
	if model.composer.Value() != "" || model.historyIndex != -1 {
		t.Fatalf("history exit = %q index=%d", model.composer.Value(), model.historyIndex)
	}
}

func TestCommandsEditorExportSwitchAndClose(t *testing.T) {
	t.Parallel()
	model := readyModel(t, nil)
	if model.SessionID() == "" {
		t.Fatal("ready model omitted session ID")
	}
	var absent *Model
	if absent.SessionID() != "" {
		t.Fatal("nil model returned a session ID")
	}
	attached := model.session
	model.session = nil
	model.snapshot.SessionID = "snapshot-session"
	if model.SessionID() != "snapshot-session" {
		t.Fatal("detached model omitted snapshot session ID")
	}
	model.session = attached
	model.command("/help")
	if !model.help {
		t.Fatal("help not shown")
	}
	model.handleKey(key(tea.KeyEscape, "", 0))
	if model.help {
		t.Fatal("escape did not hide help")
	}
	model.command("/thinking")
	if model.config.Thinking != render.ThinkingHidden {
		t.Fatalf("thinking cycle = %s", model.config.Thinking)
	}
	model.command("/tools")
	if !model.config.ToolDetails {
		t.Fatal("tool details did not toggle")
	}
	model.Update(key('t', "t", tea.ModCtrl))
	model.Update(key('g', "g", tea.ModCtrl))
	if model.config.Thinking != render.ThinkingVisible || model.config.ToolDetails {
		t.Fatalf("key toggles thinking=%s tools=%t", model.config.Thinking, model.config.ToolDetails)
	}
	model.snapshot.Transcript = []uit.Entry{{Text: "old"}}
	model.command("/clear")
	if model.clearBefore != 1 {
		t.Fatalf("clearBefore = %d", model.clearBefore)
	}
	model.command("/bogus")
	if !model.failure || !strings.Contains(model.banner, "Unknown") {
		t.Fatalf("unknown banner = %q", model.banner)
	}
	model.command("/resume")
	if !strings.Contains(model.banner, "Usage") {
		t.Fatalf("resume usage = %q", model.banner)
	}
	model.composer.SetValue("draft")
	_, noEditor := model.Update(key('e', "e", tea.ModCtrl))
	if noEditor != nil || !model.failure {
		t.Fatal("missing editor behavior incorrect")
	}
	model.editor = func(context.Context, string) (string, error) { return "edited", nil }
	_, editor := model.Update(key('e', "e", tea.ModCtrl))
	execute(t, model, editor)
	if model.composer.Value() != "edited" {
		t.Fatalf("editor value = %q", model.composer.Value())
	}
	model.editor = func(context.Context, string) (string, error) { return "", errors.New("editor") }
	_, editor = model.Update(key('e', "e", tea.ModCtrl))
	execute(t, model, editor)
	if !strings.Contains(model.banner, "editor") {
		t.Fatalf("editor error = %q", model.banner)
	}
	export := model.command("/export")
	execute(t, model, export)
	if model.LastExport() == "" {
		t.Fatalf("export = %q", model.LastExport())
	}
	oldID := model.session.ID()
	newSession := model.command("/new")
	execute(t, model, newSession)
	if model.session.ID() == oldID {
		t.Fatal("/new retained session")
	}
	resume := model.command("/resume " + oldID)
	execute(t, model, resume)
	if model.session.ID() != oldID {
		t.Fatalf("resume ID = %s", model.session.ID())
	}
	current, currentBridge := model.session, model.bridge
	execute(t, model, model.command("/resume missing"))
	if model.session != current || model.bridge != currentBridge || !model.failure {
		t.Fatal("failed resume discarded the current session")
	}
	quit := model.command("/quit")
	message := quit()
	_, quitCommand := model.Update(message)
	if quitCommand == nil || !model.closing {
		t.Fatal("quit did not close")
	}
}

func TestPermissionOverlayAndResolution(t *testing.T) {
	t.Parallel()
	model := readyModel(t, nil)
	model.composer.SetValue("need permission")
	_, command := model.Update(key(tea.KeyEnter, "", 0))
	execute(t, model, command)
	if model.snapshot.Suspension == nil || model.snapshot.State != uit.StateSuspended {
		t.Fatalf("suspension = %#v", model.snapshot)
	}
	model.resize(20, 8)
	narrow := model.View().Content
	lines := strings.Split(narrow, "\n")
	if len(lines) > 8 {
		t.Fatalf("narrow approval uses %d rows: %q", len(lines), narrow)
	}
	for _, line := range lines {
		if len([]rune(line)) > 20 {
			t.Fatalf("narrow approval line exceeds width: %q", line)
		}
	}
	model.handleApprovalKey("enter")
	if !model.failure {
		t.Fatal("incomplete approval had no failure")
	}
	model.handleApprovalKey("up")
	model.handleApprovalKey("down")
	model.handleApprovalKey("tab")
	model.handleApprovalKey("esc")
	model.handleApprovalKey("a")
	resolve := model.handleApprovalKey("enter")
	execute(t, model, resolve)
	if model.snapshot.Suspension != nil || model.snapshot.State != uit.StateIdle {
		t.Fatalf("resolved snapshot = %#v", model.snapshot)
	}
	model.snapshot.Suspension = &uit.Suspension{Kind: "custom", Description: "handoff"}
	model.handleApprovalKey("a")
	model.handleApprovalKey("esc")
	if !strings.Contains(model.banner, "handoff") {
		t.Fatalf("unsupported banner = %q", model.banner)
	}
}

func TestEventReductionRenderingAndHelpers(t *testing.T) {
	t.Parallel()
	model := readyModel(t, nil)
	entry := uit.Entry{Role: uit.RoleAssistant, Text: "committed"}
	thinking := uit.Thinking{Text: "thought"}
	tool := uit.Tool{CallID: "one", Name: "tool", State: uit.ToolRunning}
	usage := uit.Usage{TotalTokens: 10}
	queue := uit.QueuedInput{ID: "q", Text: "queued"}
	suspension := uit.Suspension{ID: "s", Kind: "permission", Supported: true}
	events := []uit.Event{
		{Cursor: 2, State: uit.StateRunning, Kind: uit.EventTextDelta, TextDelta: "hello", Usage: &usage},
		{Kind: uit.EventThinkingDelta, Thinking: &thinking},
		{Kind: uit.EventToolPlanned, Tools: []uit.Tool{tool}},
		{Kind: uit.EventToolStarted, Tool: &tool},
		{Kind: uit.EventToolResult, Tool: &uit.Tool{CallID: "one", State: uit.ToolDone}},
		{Kind: uit.EventAssistantCommitted, Entry: &entry},
		{Kind: uit.EventMessagesInjected, Entries: []uit.Entry{{Role: uit.RoleUser, Text: "injected"}}},
		{Kind: uit.EventQueueAccepted, Queue: &queue},
		{Kind: uit.EventQueueDrained, Queue: &queue},
		{Kind: uit.EventQueueAccepted, Queue: &queue},
		{Kind: uit.EventQueueCancelled, Queue: &queue},
		{Kind: uit.EventRunStarted},
		{Kind: uit.EventRunSuspended, Suspension: &suspension},
		{Kind: uit.EventRunCompleted},
		{Kind: uit.EventRunFailed, Failure: "failed"},
		{Kind: uit.EventSessionFaulted},
		{Kind: uit.EventSessionClosed},
	}
	for _, event := range events {
		model.applyEvent(event)
	}
	if model.snapshot.Cursor != 2 || len(model.snapshot.Transcript) < 2 || model.snapshot.State != uit.StateClosed || !model.failure {
		t.Fatalf("reduced model = %#v", model)
	}
	model.applyBridge(bridgeUpdate{snapshot: &uit.Snapshot{SessionID: "fresh", State: uit.StateIdle}})
	model.applyBridge(bridgeUpdate{events: []uit.Event{{Kind: uit.EventRunInterrupted}}})
	model.applyBridge(bridgeUpdate{err: errors.New("bridge")})
	model.resize(1, 1)
	view := model.View()
	if !strings.Contains(view.Content, "bridge") || view.AltScreen {
		t.Fatalf("view = %#v", view)
	}
	if got := upsertTool(nil, tool); len(got) != 1 {
		t.Fatalf("upsert append = %#v", got)
	}
	if got := upsertTool([]uit.Tool{tool}, uit.Tool{CallID: "one", State: uit.ToolDone}); got[0].State != uit.ToolDone {
		t.Fatalf("upsert replace = %#v", got)
	}
	if got := removeQueue([]uit.QueuedInput{{ID: "a"}, {ID: "b"}}, "a"); len(got) != 1 || got[0].ID != "b" {
		t.Fatalf("remove queue = %#v", got)
	}
}

func TestBridgeReconciliationDropsBufferedEventsAlreadyInSnapshot(t *testing.T) {
	t.Parallel()
	model := readyModel(t, nil)
	committed := uit.Entry{Role: uit.RoleAssistant, Text: "committed once"}
	model.snapshot = uit.Snapshot{Cursor: 2, State: uit.StateIdle, Transcript: []uit.Entry{committed}}
	model.applyBridge(bridgeUpdate{events: []uit.Event{
		{Kind: uit.EventTextDelta, TextDelta: "stale preview"},
		{Cursor: 2, Durable: true, Kind: uit.EventAssistantCommitted, Entry: &committed},
		{Kind: uit.EventTextDelta, TextDelta: "new preview"},
	}})
	if len(model.snapshot.Transcript) != 1 || model.snapshot.Transcript[0].Text != "committed once" {
		t.Fatalf("reconciled transcript = %#v", model.snapshot.Transcript)
	}
	if model.preview != "new preview" {
		t.Fatalf("preview = %q", model.preview)
	}

	next := uit.Entry{Role: uit.RoleAssistant, Text: "next"}
	model.applyBridge(bridgeUpdate{events: []uit.Event{
		{Cursor: 2, Durable: true, Kind: uit.EventRunEnded},
		{Cursor: 3, Durable: true, Kind: uit.EventAssistantCommitted, Entry: &next},
	}})
	if model.snapshot.Cursor != 3 || len(model.snapshot.Transcript) != 2 || model.snapshot.Transcript[1].Text != "next" {
		t.Fatalf("new durable event = %#v", model.snapshot)
	}
}

func TestTerminalLayoutSnapshotsAtNarrowNormalAndWideSizes(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name          string
		width, height int
		want          string
		wantSHA       string
	}{
		{name: "narrow", width: 20, height: 8, want: "20x8 viewport=20x1 composer=16x3", wantSHA: "e4d9c87cd79c8ca8b1286d8be8cdfc49ce5a8d264ee5fc5886d17f4e75be4725"},
		{name: "normal", width: 80, height: 24, want: "80x24 viewport=80x16 composer=76x4", wantSHA: "30bba916c17f8852244c4f024a1f6b18742faeb65a030a09474f5ae9e3c74e19"},
		{name: "wide", width: 160, height: 50, want: "160x50 viewport=160x41 composer=156x5", wantSHA: "f178eeb4f2277fa8c31859a4d8d3a12dd5c16e1e3fabf0033e7d8c5c6770f717"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := readyModel(t, nil)
			model.snapshot = uit.Snapshot{
				SessionID: "layout", State: uit.StateIdle, ProfileLabel: "offline",
				Transcript: []uit.Entry{{Role: uit.RoleUser, Text: "deterministic layout probe"}},
			}
			model.resize(test.width, test.height)
			got := fmt.Sprintf(
				"%dx%d viewport=%dx%d composer=%dx%d",
				model.width, model.height,
				model.viewport.Width(), model.viewport.Height(),
				model.composer.Width(), model.composer.Height(),
			)
			if got != test.want {
				t.Fatalf("layout snapshot = %q, want %q", got, test.want)
			}
			view := model.View()
			content := view.Content
			if gotSHA := fmt.Sprintf("%x", sha256.Sum256([]byte(content))); gotSHA != test.wantSHA {
				t.Fatalf("view snapshot hash = %s, want %s\n%s", gotSHA, test.wantSHA, content)
			}
			if strings.Contains(content, "\x1b") || view.Cursor == nil || view.Cursor.X < 0 || view.Cursor.Y < 0 || view.Cursor.Y >= model.height {
				t.Fatalf("no-color cursor/view is not terminal-safe: cursor=%#v content=%q", view.Cursor, content)
			}
			lines := strings.Split(content, "\n")
			if len(lines) > test.height {
				t.Fatalf("view has %d rows at height %d", len(lines), test.height)
			}
			for _, line := range lines {
				if len([]rune(line)) > test.width {
					t.Fatalf("view line exceeds width %d: %q", test.width, line)
				}
			}
			for _, stable := range []string{"probe", "session layout", "enter send"} {
				if !strings.Contains(content, stable) {
					t.Fatalf("view at %s omitted %q: %q", test.name, stable, content)
				}
			}
		})
	}
}

func TestSessionOpenAndOperationFailures(t *testing.T) {
	t.Parallel()
	want := errors.New("host")
	host := failingHost{err: want}
	model, err := New(host, Options{Config: DefaultConfig()})
	if err != nil {
		t.Fatal(err)
	}
	message := model.Init()()
	model.Update(message)
	if !strings.Contains(model.banner, "host") {
		t.Fatalf("open error = %q", model.banner)
	}
	model.Update(tea.InterruptMsg{})
	model.cancel()

	ready := readyModel(t, nil)
	ready.composer.SetValue("/unknown")
	ready.Update(key(tea.KeyEnter, "", 0))
	ready.session = nil
	ready.composer.SetValue("task")
	ready.dispatchDraft("submit")
	if !ready.failure {
		t.Fatal("nil session submit had no failure")
	}
}

func TestReducerDefensiveBranchesAndRunValidation(t *testing.T) {
	t.Parallel()
	model := readyModel(t, nil)
	if message := waitBridgeCmd(nil)(); message.(bridgeMsg).ok {
		t.Fatal("nil bridge wait reported an event")
	}
	other := &eventBridge{}
	model.Update(bridgeMsg{bridge: other, ok: true})
	model.Update(bridgeMsg{bridge: model.bridge, ok: false})
	model.Update(bridgeMsg{bridge: model.bridge, ok: true, update: bridgeUpdate{events: []uit.Event{{Kind: uit.EventTextDelta, TextDelta: "live"}}}})
	model.Update(operationDoneMsg{err: errors.New("operation")})
	model.Update(editorDoneMsg{err: errors.New("editor")})
	if !model.failure {
		t.Fatal("operation/editor errors were not displayed")
	}
	model.clearBefore = 100
	model.syncViewport()
	if model.clearBefore != len(model.snapshot.Transcript) {
		t.Fatal("clear boundary was not clamped")
	}
	model.help = true
	if !strings.Contains(model.View().Content, "Commands:") {
		t.Fatal("help view missing")
	}
	model.help = false
	model.snapshot.Suspension = &uit.Suspension{Kind: "custom", Description: "handoff"}
	if !strings.Contains(model.View().Content, "handoff") {
		t.Fatal("suspension view missing")
	}
	model.snapshot.Suspension = nil
	model.composer.Reset()
	if model.dispatchDraft("submit") != nil {
		t.Fatal("empty draft produced a command")
	}
	model.host = basicHost{session: model.session}
	execute(t, model, model.command("/export"))
	if model.LastExport() == "" || model.failure {
		t.Fatalf("fallback export = %q banner=%q", model.LastExport(), model.banner)
	}
	model.handleApprovalKey("a")
	model.snapshot.Suspension = &uit.Suspension{ID: "s", Supported: true, Approvals: []uit.Approval{{CallID: "c"}}}
	model.handleApprovalKey("d")
	if model.decisions["c"] != uit.DecisionDeny {
		t.Fatal("deny decision missing")
	}
	model.snapshot.Suspension = nil
	model.applyEvent(uit.Event{Kind: uit.EventToolPlanned, Tool: &uit.Tool{CallID: "new", State: uit.ToolPreview}})
	model.thinking = "thinking"
	model.syncViewport()
	model.Update(key('c', "c", tea.ModCtrl))
	model.session = nil
	if command := model.interruptCmd(); command != nil || !model.failure {
		t.Fatal("nil-session interrupt behavior incorrect")
	}
	if _, err := Run(context.Background(), nil, Options{}); err == nil {
		t.Fatal("Run with nil host succeeded")
	}
}

func TestSessionAndOperationFailureWorkers(t *testing.T) {
	t.Parallel()
	want := errors.New("snapshot")
	session := &errorSession{id: "error", snapshotErr: want}
	message := openSessionCmd(context.Background(), basicHost{session: session}, "", DefaultConfig())()
	if ready := message.(sessionReadyMsg); !errors.Is(ready.err, want) || session.closeCalls != 1 {
		t.Fatalf("open failure = %#v closes=%d", ready, session.closeCalls)
	}
	if resumed := openSessionCmd(context.Background(), basicHost{session: session}, "id", DefaultConfig())().(sessionReadyMsg); !errors.Is(resumed.err, want) {
		t.Fatalf("resume failure = %#v", resumed)
	}

	model := readyModel(t, nil)
	model.session = &errorSession{id: "operation", snapshotErr: want}
	result := model.operationCmd(func(context.Context) error { return nil }, "ok")().(operationDoneMsg)
	if !errors.Is(result.err, want) {
		t.Fatalf("snapshot operation error = %v", result.err)
	}
	wantOperation := errors.New("operation")
	result = model.operationCmd(func(context.Context) error { return wantOperation }, "ok")().(operationDoneMsg)
	if !errors.Is(result.err, wantOperation) {
		t.Fatalf("operation error = %v", result.err)
	}
	closing := &errorSession{id: "closing", closeErr: errors.New("close")}
	model.session, model.bridge = closing, nil
	if result := model.closeCmd()().(operationDoneMsg); result.err == nil {
		t.Fatal("close error missing")
	}
	if ready := model.switchSessionCmd("")().(sessionReadyMsg); ready.err == nil {
		t.Fatal("switch close error missing")
	}
	model.session = closing
	_, interrupt := model.Update(tea.InterruptMsg{})
	execute(t, model, interrupt)
}

func TestRunOfflineProgramToCleanQuit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	final, err := Run(ctx, testhost.New(nil), Options{
		Config:  Config{AlternateScreen: AlternateNever, NoColor: true, Thinking: render.ThinkingCollapsed, PreviewHz: 60, EventBuffer: 32},
		Environ: []string{"TERM=dumb"},
	}, tea.WithInput(strings.NewReader("/quit\r")), tea.WithOutput(&bytes.Buffer{}), tea.WithEnvironment([]string{"TERM=dumb"}), tea.WithoutRenderer())
	if err != nil {
		t.Fatal(err)
	}
	if final == nil || !final.closing {
		t.Fatalf("final model = %#v", final)
	}
}

type failingHost struct{ err error }

func (h failingHost) NewSession(context.Context, uit.SessionOptions) (uit.Session, error) {
	return nil, h.err
}
func (h failingHost) ResumeSession(context.Context, string) (uit.Session, error) {
	return nil, h.err
}

type basicHost struct{ session uit.Session }

func (h basicHost) NewSession(context.Context, uit.SessionOptions) (uit.Session, error) {
	return h.session, nil
}
func (h basicHost) ResumeSession(context.Context, string) (uit.Session, error) { return h.session, nil }

type errorSession struct {
	id          string
	snapshot    uit.Snapshot
	snapshotErr error
	closeErr    error
	closeCalls  int
}

func (s *errorSession) ID() string { return s.id }
func (s *errorSession) Snapshot(context.Context) (uit.Snapshot, error) {
	return s.snapshot, s.snapshotErr
}
func (s *errorSession) Subscribe(uit.SubscribeOptions) uit.Subscription { return nil }
func (s *errorSession) Submit(context.Context, uit.Input) error         { return nil }
func (s *errorSession) Steer(context.Context, uit.Input) error          { return nil }
func (s *errorSession) FollowUp(context.Context, uit.Input) error       { return nil }
func (s *errorSession) NextTurn(context.Context, uit.Input) error       { return nil }
func (s *errorSession) Resolve(context.Context, uit.Resolution) error   { return nil }
func (s *errorSession) Interrupt(context.Context) error                 { return errors.New("interrupt") }
func (s *errorSession) Close(context.Context) error {
	s.closeCalls++
	return s.closeErr
}
