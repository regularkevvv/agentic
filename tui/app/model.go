// Package app implements the Bubble Tea terminal client. Blocking session
// operations and subscriptions execute in commands or bridge goroutines,
// never in Model.Update.
package app

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"unicode/utf8"

	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"

	uit "github.com/regularkevvv/agentic/tui"
	"github.com/regularkevvv/agentic/tui/render"
)

type AlternateScreen string

const (
	AlternateAuto   AlternateScreen = "auto"
	AlternateAlways AlternateScreen = "always"
	AlternateNever  AlternateScreen = "never"
)

type Config struct {
	AlternateScreen AlternateScreen
	NoColor         bool
	Thinking        render.ThinkingMode
	ToolDetails     bool
	PreviewHz       int
	EventBuffer     int
}

func DefaultConfig() Config {
	_, noColor := os.LookupEnv("NO_COLOR")
	return Config{
		AlternateScreen: AlternateAuto, NoColor: noColor,
		Thinking: render.ThinkingCollapsed, PreviewHz: 60, EventBuffer: 256,
	}
}

func (c Config) Validate() error {
	if c.AlternateScreen != AlternateAuto && c.AlternateScreen != AlternateAlways && c.AlternateScreen != AlternateNever {
		return fmt.Errorf("invalid alternate-screen mode %q", c.AlternateScreen)
	}
	if c.Thinking != render.ThinkingVisible && c.Thinking != render.ThinkingCollapsed && c.Thinking != render.ThinkingHidden {
		return fmt.Errorf("invalid thinking mode %q", c.Thinking)
	}
	if c.PreviewHz < 0 || c.PreviewHz > 240 {
		return errors.New("preview rate must be zero or between 1 and 240 Hz")
	}
	if c.EventBuffer < 0 {
		return errors.New("event buffer must not be negative")
	}
	return nil
}

type Editor func(context.Context, string) (string, error)

type TerminalEditor interface {
	Prepare(context.Context, string) (*exec.Cmd, func() (string, error), error)
}

type Options struct {
	Config         Config
	ResumeID       string
	Context        context.Context
	Editor         Editor
	TerminalEditor TerminalEditor
	Environ        []string
}

type Model struct {
	host           uit.Host
	session        uit.Session
	snapshot       uit.Snapshot
	bridge         *eventBridge
	ctx            context.Context
	cancel         context.CancelFunc
	config         Config
	resumeID       string
	editor         Editor
	terminalEditor TerminalEditor
	environ        []string

	composer            textarea.Model
	viewport            viewport.Model
	followOutput        bool
	width               int
	height              int
	preview             string
	thinking            string
	liveTools           []uit.Tool
	clearBefore         int
	inflight            int
	banner              string
	failure             bool
	help                bool
	closing             bool
	approvalIndex       int
	decisions           map[string]uit.DecisionAction
	resolvingApprovalID string
	optimistic          []optimisticSubmission
	nextOptimisticID    uint64
	history             []string
	historyIndex        int
	lastExport          string
}

func New(host uit.Host, options Options) (*Model, error) {
	if host == nil {
		return nil, errors.New("TUI app requires a host")
	}
	config := options.Config
	if config == (Config{}) {
		config = DefaultConfig()
	}
	if config.PreviewHz == 0 {
		config.PreviewHz = 60
	}
	if config.EventBuffer == 0 {
		config.EventBuffer = 256
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	ctx := options.Context
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	environ := options.Environ
	if environ == nil {
		environ = os.Environ()
	}
	for _, value := range environ {
		if value == "NO_COLOR" || strings.HasPrefix(value, "NO_COLOR=") {
			config.NoColor = true
		}
	}
	composer := textarea.New()
	composer.Placeholder = "Describe a task…"
	composer.ShowLineNumbers = false
	composer.CharLimit = 1 << 20
	composer.MaxHeight = 8
	composer.MinHeight = 1
	composer.DynamicHeight = true
	composer.SetHeight(1)
	composer.SetWidth(80)
	if config.NoColor {
		composer.Placeholder = "Describe a task..."
		composer.Prompt = "> "
		composer.SetStyles(textarea.Styles{})
		composer.SetVirtualCursor(false)
	}
	composer.Focus()
	vp := viewport.New(viewport.WithWidth(80), viewport.WithHeight(16))
	vp.SoftWrap = true
	return &Model{
		host: host, ctx: ctx, cancel: cancel, config: config, resumeID: options.ResumeID,
		editor: options.Editor, terminalEditor: options.TerminalEditor, environ: environ, composer: composer, viewport: vp, followOutput: true,
		decisions: make(map[string]uit.DecisionAction), historyIndex: -1,
	}, nil
}

func (m *Model) Init() tea.Cmd { return openSessionCmd(m.ctx, m.host, m.resumeID, m.config) }

type sessionReadyMsg struct {
	session  uit.Session
	snapshot uit.Snapshot
	bridge   *eventBridge
	err      error
}

type operationDoneMsg struct {
	snapshot         *uit.Snapshot
	message          string
	export           string
	err              error
	quit             bool
	approval         bool
	optimisticID     uint64
	operationApplied bool
}

type optimisticSubmission struct {
	id    uint64
	index int
	entry uit.Entry
}

type bridgeMsg struct {
	bridge *eventBridge
	update bridgeUpdate
	ok     bool
}

type editorDoneMsg struct {
	value string
	err   error
}

type editorPreparedMsg struct {
	command *exec.Cmd
	read    func() (string, error)
	err     error
}

type editorExecutedMsg struct {
	read func() (string, error)
	err  error
}

func openSessionCmd(ctx context.Context, host uit.Host, id string, config Config) tea.Cmd {
	return func() tea.Msg {
		var session uit.Session
		var err error
		if id == "" {
			session, err = host.NewSession(ctx, uit.SessionOptions{})
		} else {
			session, err = host.ResumeSession(ctx, id)
		}
		if err != nil {
			return sessionReadyMsg{err: err}
		}
		snapshot, err := session.Snapshot(ctx)
		if err != nil {
			_ = session.Close(ctx)
			return sessionReadyMsg{err: err}
		}
		return sessionReadyMsg{session: session, snapshot: snapshot, bridge: startBridge(ctx, session, snapshot, config.PreviewHz, config.EventBuffer)}
	}
}

func waitBridgeCmd(bridge *eventBridge) tea.Cmd {
	return func() tea.Msg {
		if bridge == nil {
			return bridgeMsg{}
		}
		update, ok := <-bridge.updates
		return bridgeMsg{bridge: bridge, update: update, ok: ok}
	}
}

func (m *Model) Update(message tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := message.(type) {
	case sessionReadyMsg:
		m.inflight = max(0, m.inflight-1)
		if msg.err != nil {
			m.setError(msg.err)
			return m, nil
		}
		m.replaceSession(msg.session, msg.snapshot, msg.bridge)
		return m, waitBridgeCmd(msg.bridge)
	case bridgeMsg:
		if msg.bridge != m.bridge {
			return m, nil
		}
		if !msg.ok {
			return m, nil
		}
		m.applyBridge(msg.update)
		return m, waitBridgeCmd(msg.bridge)
	case operationDoneMsg:
		m.inflight = max(0, m.inflight-1)
		if msg.approval {
			m.resolvingApprovalID = ""
		}
		if msg.snapshot != nil {
			m.snapshot = *msg.snapshot
			m.reconcileOptimistic()
			m.preview, m.thinking, m.liveTools = "", "", nil
			m.resetApproval()
		}
		if msg.err != nil {
			if msg.optimisticID != 0 && !msg.operationApplied {
				m.rollbackOptimistic(msg.optimisticID)
			}
			m.setError(msg.err)
		} else {
			if msg.message != "" {
				m.banner, m.failure = msg.message, false
			}
			if msg.export != "" {
				m.lastExport = msg.export
			}
		}
		m.syncViewport()
		if msg.quit {
			m.cancel()
			return m, tea.Quit
		}
		return m, nil
	case editorDoneMsg:
		m.inflight = max(0, m.inflight-1)
		if msg.err != nil {
			m.setError(msg.err)
		} else {
			m.composer.SetValue(msg.value)
			m.syncViewport()
		}
		return m, nil
	case editorPreparedMsg:
		if msg.err != nil {
			m.inflight = max(0, m.inflight-1)
			m.setError(msg.err)
			return m, nil
		}
		return m, tea.ExecProcess(msg.command, func(err error) tea.Msg { return editorExecutedMsg{read: msg.read, err: err} })
	case editorExecutedMsg:
		if msg.err != nil {
			m.inflight = max(0, m.inflight-1)
			m.setError(msg.err)
			return m, nil
		}
		return m, func() tea.Msg {
			value, err := msg.read()
			return editorDoneMsg{value: value, err: err}
		}
	case tea.WindowSizeMsg:
		m.resize(msg.Width, msg.Height)
		return m, nil
	case tea.InterruptMsg:
		return m, m.interruptCmd()
	case tea.KeyPressMsg:
		if command, handled := m.handleKey(msg); handled {
			return m, command
		}
	}

	var commands []tea.Cmd
	var command tea.Cmd
	composerHeight := m.composer.Height()
	m.composer, command = m.composer.Update(message)
	commands = append(commands, command)
	if m.composer.Height() != composerHeight {
		m.syncViewport()
	}
	if _, ok := message.(tea.MouseWheelMsg); ok {
		command = m.updateViewport(message)
		commands = append(commands, command)
	}
	return m, tea.Batch(commands...)
}

func (m *Model) handleKey(key tea.KeyPressMsg) (tea.Cmd, bool) {
	stroke := key.Keystroke()
	if m.resolvingApprovalID != "" {
		if stroke == "ctrl+c" {
			return m.interruptCmd(), true
		}
		return nil, true
	}
	if m.snapshot.Suspension != nil {
		return m.handleApprovalKey(stroke), true
	}
	switch stroke {
	case "ctrl+c":
		return m.interruptCmd(), true
	case "ctrl+e":
		if m.terminalEditor != nil {
			m.inflight++
			value := m.composer.Value()
			return func() tea.Msg {
				command, read, err := m.terminalEditor.Prepare(m.ctx, value)
				return editorPreparedMsg{command: command, read: read, err: err}
			}, true
		}
		if m.editor == nil {
			m.banner, m.failure = "No external editor is configured.", true
			return nil, true
		}
		m.inflight++
		value := m.composer.Value()
		return func() tea.Msg {
			result, err := m.editor(m.ctx, value)
			return editorDoneMsg{value: result, err: err}
		}, true
	case "ctrl+t":
		m.cycleThinking()
		return nil, true
	case "ctrl+g":
		m.config.ToolDetails = !m.config.ToolDetails
		m.banner, m.failure = "Tool details toggled.", false
		m.syncViewport()
		return nil, true
	case "pgup", "pgdown", "ctrl+u", "ctrl+d":
		return m.updateViewport(key), true
	case "enter":
		return m.dispatchDraft("submit"), true
	case "shift+enter", "ctrl+j":
		m.composer.InsertString("\n")
		m.syncViewport()
		return nil, true
	case "alt+s":
		return m.dispatchDraft("steer"), true
	case "alt+f":
		return m.dispatchDraft("follow-up"), true
	case "alt+n":
		return m.dispatchDraft("next-turn"), true
	case "up":
		if len(m.history) > 0 && (m.historyIndex >= 0 || m.composer.Value() == "") {
			if m.historyIndex < 0 {
				m.historyIndex = len(m.history) - 1
			} else if m.historyIndex > 0 {
				m.historyIndex--
			}
			m.composer.SetValue(m.history[m.historyIndex])
			m.syncViewport()
			return nil, true
		}
	case "down":
		if m.historyIndex >= 0 {
			if m.historyIndex < len(m.history)-1 {
				m.historyIndex++
				m.composer.SetValue(m.history[m.historyIndex])
			} else {
				m.historyIndex = -1
				m.composer.Reset()
			}
			m.syncViewport()
			return nil, true
		}
	case "esc":
		m.help = false
		return nil, true
	}
	return nil, false
}

func (m *Model) dispatchDraft(action string) tea.Cmd {
	text := m.composer.Value()
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if strings.HasPrefix(strings.TrimSpace(text), "/") {
		m.composer.Reset()
		m.syncViewport()
		return m.command(strings.TrimSpace(text))
	}
	if m.session == nil {
		m.banner, m.failure = "Session is not ready.", true
		return nil
	}
	input := uit.Input{Text: text}
	session := m.session
	m.composer.Reset()
	m.history = append(m.history, text)
	m.historyIndex = -1
	m.inflight++
	success := action + " accepted"
	var optimisticID uint64
	if action == "submit" {
		success = ""
		optimisticID = m.appendOptimistic(input.Text)
		m.followOutput = true
	}
	m.syncViewport()
	return m.operationCmd(func(ctx context.Context) error {
		switch action {
		case "steer":
			return session.Steer(ctx, input)
		case "follow-up":
			return session.FollowUp(ctx, input)
		case "next-turn":
			return session.NextTurn(ctx, input)
		default:
			return session.Submit(ctx, input)
		}
	}, success, optimisticID)
}

func (m *Model) operationCmd(operation func(context.Context) error, success string, optimisticIDs ...uint64) tea.Cmd {
	session := m.session
	var optimisticID uint64
	var optimisticEntry *uit.Entry
	if len(optimisticIDs) > 0 {
		optimisticID = optimisticIDs[0]
		for _, pending := range m.optimistic {
			if pending.id == optimisticID {
				entry := pending.entry
				optimisticEntry = &entry
				break
			}
		}
	}
	return func() tea.Msg {
		operationErr := operation(m.ctx)
		snapshot, snapshotErr := session.Snapshot(m.ctx)
		if snapshotErr != nil {
			return operationDoneMsg{err: errors.Join(operationErr, snapshotErr), optimisticID: optimisticID}
		}
		applied := optimisticEntry == nil || transcriptContains(snapshot.Transcript, *optimisticEntry)
		return operationDoneMsg{
			snapshot: &snapshot, message: success, err: operationErr,
			optimisticID: optimisticID, operationApplied: applied,
		}
	}
}

func transcriptContains(entries []uit.Entry, want uit.Entry) bool {
	for _, entry := range entries {
		if sameInputEntry(entry, want) {
			return true
		}
	}
	return false
}

func (m *Model) appendOptimistic(text string) uint64 {
	m.nextOptimisticID++
	entry := uit.Entry{Role: uit.RoleUser, Text: text}
	pending := optimisticSubmission{id: m.nextOptimisticID, index: len(m.snapshot.Transcript), entry: entry}
	m.snapshot.Transcript = append(m.snapshot.Transcript, entry)
	m.optimistic = append(m.optimistic, pending)
	return pending.id
}

func (m *Model) confirmOptimistic(entry uit.Entry) bool {
	for index, pending := range m.optimistic {
		if !sameInputEntry(pending.entry, entry) || pending.index >= len(m.snapshot.Transcript) || !sameInputEntry(m.snapshot.Transcript[pending.index], entry) {
			continue
		}
		m.optimistic = append(m.optimistic[:index], m.optimistic[index+1:]...)
		return true
	}
	return false
}

func (m *Model) reconcileOptimistic() {
	remaining := make([]optimisticSubmission, 0, len(m.optimistic))
	for _, pending := range m.optimistic {
		start := min(pending.index, len(m.snapshot.Transcript))
		found := false
		for _, entry := range m.snapshot.Transcript[start:] {
			if sameInputEntry(pending.entry, entry) {
				found = true
				break
			}
		}
		if found {
			continue
		}
		pending.index = len(m.snapshot.Transcript)
		m.snapshot.Transcript = append(m.snapshot.Transcript, pending.entry)
		remaining = append(remaining, pending)
	}
	m.optimistic = remaining
}

func (m *Model) rollbackOptimistic(id uint64) {
	for optimisticIndex, pending := range m.optimistic {
		if pending.id != id {
			continue
		}
		if pending.index < len(m.snapshot.Transcript) && sameInputEntry(m.snapshot.Transcript[pending.index], pending.entry) {
			m.snapshot.Transcript = append(m.snapshot.Transcript[:pending.index], m.snapshot.Transcript[pending.index+1:]...)
			for index := range m.optimistic {
				if m.optimistic[index].index > pending.index {
					m.optimistic[index].index--
				}
			}
		}
		m.optimistic = append(m.optimistic[:optimisticIndex], m.optimistic[optimisticIndex+1:]...)
		return
	}
}

func sameInputEntry(left, right uit.Entry) bool {
	return left.Role == uit.RoleUser && right.Role == uit.RoleUser && left.Text == right.Text
}

func (m *Model) command(value string) tea.Cmd {
	fields := strings.Fields(value)
	name := fields[0]
	switch name {
	case "/help":
		m.help = !m.help
		return nil
	case "/thinking":
		m.cycleThinking()
		return nil
	case "/tools":
		m.config.ToolDetails = !m.config.ToolDetails
		m.banner, m.failure = "Tool details toggled.", false
		m.syncViewport()
		return nil
	case "/clear":
		m.clearBefore = len(m.snapshot.Transcript)
		m.preview = ""
		m.syncViewport()
		return nil
	case "/quit":
		m.closing = true
		m.inflight++
		return m.closeCmd()
	case "/new":
		m.inflight++
		return m.switchSessionCmd("")
	case "/resume":
		if len(fields) != 2 {
			m.banner, m.failure = "Usage: /resume SESSION_ID", true
			return nil
		}
		m.inflight++
		return m.switchSessionCmd(fields[1])
	case "/export":
		if m.session == nil {
			m.banner, m.failure = "Session is not ready.", true
			return nil
		}
		m.inflight++
		exporter, custom := m.host.(uit.TranscriptExporter)
		id, snapshot, options := m.session.ID(), m.snapshot, m.config
		return func() tea.Msg {
			if custom {
				value, err := exporter.ExportTranscript(m.ctx, id)
				return operationDoneMsg{message: "Transcript exported in memory.", export: value, err: err}
			}
			value := render.Transcript(snapshot.Transcript, "", render.Options{
				NoColor: true, Thinking: options.Thinking, ToolExpanded: options.ToolDetails,
			}) + "\n"
			return operationDoneMsg{message: "Redacted transcript exported in memory.", export: value}
		}
	default:
		m.banner, m.failure = "Unknown command "+name+". Use /help.", true
		return nil
	}
}

func (m *Model) switchSessionCmd(id string) tea.Cmd {
	old, oldBridge := m.session, m.bridge
	return func() tea.Msg {
		ready, ok := openSessionCmd(m.ctx, m.host, id, m.config)().(sessionReadyMsg)
		if !ok {
			return sessionReadyMsg{err: errors.New("session opener returned an invalid message")}
		}
		if ready.err != nil {
			return ready
		}
		if old != nil {
			if err := old.Close(m.ctx); err != nil {
				if ready.bridge != nil {
					ready.bridge.Close()
				}
				if ready.session != nil {
					_ = ready.session.Close(context.WithoutCancel(m.ctx))
				}
				return sessionReadyMsg{err: err}
			}
		}
		if oldBridge != nil {
			oldBridge.Close()
		}
		return ready
	}
}

func (m *Model) interruptCmd() tea.Cmd {
	if m.session == nil {
		m.banner, m.failure = "Session is not ready.", true
		return nil
	}
	m.inflight++
	return m.operationCmd(m.session.Interrupt, "Interrupt requested.")
}

func (m *Model) closeCmd() tea.Cmd {
	session, bridge := m.session, m.bridge
	return func() tea.Msg {
		if session != nil {
			if err := session.Close(m.ctx); err != nil {
				return operationDoneMsg{err: err}
			}
		}
		if bridge != nil {
			bridge.Close()
		}
		return operationDoneMsg{quit: true}
	}
}

func (m *Model) handleApprovalKey(stroke string) tea.Cmd {
	suspension := m.snapshot.Suspension
	if suspension == nil {
		return nil
	}
	if !suspension.Supported {
		if stroke == "esc" {
			m.banner, m.failure = "Suspension left unresolved for application handoff.", false
		}
		return nil
	}
	switch stroke {
	case "up":
		m.approvalIndex = max(0, m.approvalIndex-1)
	case "down", "tab":
		m.approvalIndex = min(len(suspension.Approvals)-1, m.approvalIndex+1)
	case "a", "d":
		if len(suspension.Approvals) > 0 {
			action := uit.DecisionApprove
			if stroke == "d" {
				action = uit.DecisionDeny
			}
			m.decisions[suspension.Approvals[m.approvalIndex].CallID] = action
		}
	case "enter":
		if len(m.decisions) != len(suspension.Approvals) {
			m.banner, m.failure = "Choose approve or deny for every operation.", true
			return nil
		}
		resolution := uit.Resolution{SuspensionID: suspension.ID}
		for _, approval := range suspension.Approvals {
			resolution.Decisions = append(resolution.Decisions, uit.Decision{CallID: approval.CallID, Action: m.decisions[approval.CallID]})
		}
		session := m.session
		m.resolvingApprovalID = suspension.ID
		m.banner, m.failure = "", false
		m.inflight++
		m.syncViewport()
		return m.approvalOperationCmd(func(ctx context.Context) error { return session.Resolve(ctx, resolution) })
	case "esc":
		m.banner, m.failure = "Suspension left unresolved for safe handoff.", false
	}
	return nil
}

func (m *Model) approvalOperationCmd(operation func(context.Context) error) tea.Cmd {
	session := m.session
	return func() tea.Msg {
		if err := operation(m.ctx); err != nil {
			return operationDoneMsg{err: err, approval: true}
		}
		snapshot, err := session.Snapshot(m.ctx)
		return operationDoneMsg{snapshot: &snapshot, err: err, approval: true}
	}
}

func (m *Model) applyBridge(update bridgeUpdate) {
	if update.err != nil {
		m.setError(update.err)
		return
	}
	if update.snapshot != nil {
		m.snapshot = *update.snapshot
		m.reconcileOptimistic()
		m.preview, m.thinking, m.liveTools = "", "", nil
		if m.snapshot.Suspension == nil || m.snapshot.Suspension.ID != m.resolvingApprovalID {
			m.resolvingApprovalID = ""
			m.resetApproval()
		}
	}
	// An operation worker can reconcile a newer full snapshot before the bridge
	// delivers an already-buffered preview/commit batch. Drop the complete stale
	// prefix through the last durable cursor represented by that snapshot. This
	// prevents both duplicate transcript entries and an orphaned old preview.
	staleThrough := -1
	for index, event := range update.events {
		if event.Durable && event.Cursor <= m.snapshot.Cursor {
			staleThrough = index
		}
	}
	for index, event := range update.events {
		if index <= staleThrough || event.Durable && event.Cursor <= m.snapshot.Cursor {
			continue
		}
		m.applyEvent(event)
	}
	m.syncViewport()
}

func (m *Model) applyEvent(event uit.Event) {
	if event.Cursor > m.snapshot.Cursor {
		m.snapshot.Cursor = event.Cursor
	}
	if event.State != "" {
		m.snapshot.State = event.State
	}
	if event.Usage != nil {
		m.snapshot.Usage = *event.Usage
	}
	if event.Suspension != nil {
		m.snapshot.Suspension = event.Suspension
		m.snapshot.State = uit.StateSuspended
		if event.Suspension.ID != m.resolvingApprovalID {
			m.resetApproval()
		}
	}
	if event.Failure != "" {
		m.banner, m.failure = event.Failure, true
	}
	switch event.Kind {
	case uit.EventTextDelta:
		m.preview += event.TextDelta
	case uit.EventThinkingDelta:
		if event.Thinking != nil {
			m.thinking += event.Thinking.Text
		}
	case uit.EventAssistantCommitted:
		m.preview = ""
		if event.Entry != nil {
			m.snapshot.Transcript = append(m.snapshot.Transcript, *event.Entry)
		}
	case uit.EventMessagesInjected:
		for _, entry := range event.Entries {
			if !m.confirmOptimistic(entry) {
				m.snapshot.Transcript = append(m.snapshot.Transcript, entry)
			}
		}
	case uit.EventToolPlanned:
		m.liveTools = append(m.liveTools, event.Tools...)
		if event.Tool != nil {
			m.liveTools = upsertTool(m.liveTools, *event.Tool)
		}
	case uit.EventToolStarted, uit.EventToolResult:
		if event.Tool != nil {
			m.liveTools = upsertTool(m.liveTools, *event.Tool)
		}
	case uit.EventQueueAccepted:
		if event.Queue != nil {
			m.snapshot.Pending = append(m.snapshot.Pending, *event.Queue)
		}
	case uit.EventQueueDrained, uit.EventQueueCancelled:
		if event.Queue != nil {
			m.snapshot.Pending = removeQueue(m.snapshot.Pending, event.Queue.ID)
		}
	case uit.EventRunStarted:
		m.snapshot.State = uit.StateRunning
		m.banner, m.failure = "", false
		if m.resolvingApprovalID != "" {
			m.snapshot.Suspension = nil
			m.resolvingApprovalID = ""
			m.resetApproval()
		}
	case uit.EventRunCompleted:
		m.snapshot.State = uit.StateIdle
		m.snapshot.Suspension = nil
		m.resolvingApprovalID = ""
		m.resetApproval()
		m.preview, m.thinking, m.liveTools = "", "", nil
		m.banner, m.failure = "Completed.", false
	case uit.EventRunEnded:
		m.snapshot.State = uit.StateIdle
		m.snapshot.Suspension = nil
		m.resolvingApprovalID = ""
		m.resetApproval()
		m.preview, m.thinking, m.liveTools = "", "", nil
		if !m.failure && m.banner != "Interrupted." {
			m.banner = "Completed."
		}
	case uit.EventRunInterrupted:
		m.snapshot.State = uit.StateIdle
		m.snapshot.Suspension = nil
		m.resolvingApprovalID = ""
		m.resetApproval()
		m.preview, m.thinking, m.liveTools = "", "", nil
		m.banner, m.failure = "Interrupted.", false
	case uit.EventRunFailed, uit.EventSessionFaulted:
		m.snapshot.State = uit.StateFaulted
		m.snapshot.Suspension = nil
		m.resolvingApprovalID = ""
	case uit.EventSessionClosed:
		m.snapshot.State = uit.StateClosed
	}
}

func (m *Model) replaceSession(session uit.Session, snapshot uit.Snapshot, bridge *eventBridge) {
	if m.bridge != nil && m.bridge != bridge {
		m.bridge.Close()
	}
	m.session, m.snapshot, m.bridge = session, snapshot, bridge
	m.followOutput = true
	m.optimistic = nil
	m.clearBefore = 0
	m.preview, m.thinking, m.liveTools = "", "", nil
	m.banner, m.failure = "Session "+session.ID()+" ready.", false
	m.resolvingApprovalID = ""
	m.resetApproval()
	m.syncViewport()
}

func (m *Model) resetApproval() {
	m.approvalIndex = 0
	m.decisions = make(map[string]uit.DecisionAction)
}

func (m *Model) setError(err error) {
	if err != nil {
		m.banner, m.failure = err.Error(), true
	}
}

func (m *Model) resize(width, height int) {
	m.width, m.height = max(20, width), max(8, height)
	m.composer.MaxHeight = min(8, max(1, m.height/3))
	m.composer.SetWidth(max(10, m.width-2))
	m.composer.SetHeight(min(m.composer.MaxHeight, max(1, m.composer.LineCount())))
	m.viewport.SetWidth(m.width)
	m.viewport.SetHeight(max(1, m.height-m.composer.Height()-4))
	m.syncViewport()
}

func (m *Model) syncViewport() {
	if m.height > 0 {
		bottomRows := m.composer.Height()
		if m.resolvingApprovalID != "" {
			bottomRows = 1
		} else if m.snapshot.Suspension != nil {
			bottomRows = len(m.snapshot.Suspension.Approvals) + 3
		} else if m.help {
			bottomRows = 2
		}
		m.viewport.SetHeight(max(1, m.height-bottomRows-4))
	}
	entries := m.transcriptForDisplay()
	if m.clearBefore > len(entries) {
		m.clearBefore = len(entries)
	}
	entries = entries[m.clearBefore:]
	if len(m.liveTools) > 0 || m.thinking != "" {
		entry := uit.Entry{Role: uit.RoleAssistant, Thinking: []uit.Thinking{{Text: m.thinking}}, Tools: append([]uit.Tool(nil), m.liveTools...)}
		entries = append(append([]uit.Entry(nil), entries...), entry)
	}
	m.viewport.SetContent(render.Transcript(entries, m.preview, render.Options{
		Width: m.width, NoColor: m.config.NoColor, Thinking: m.config.Thinking, ToolExpanded: m.config.ToolDetails,
	}))
	if m.followOutput {
		m.viewport.GotoBottom()
	}
}

func (m *Model) transcriptForDisplay() []uit.Entry {
	if len(m.optimistic) == 0 {
		return m.snapshot.Transcript
	}
	entries := append([]uit.Entry(nil), m.snapshot.Transcript...)
	for _, pending := range m.optimistic {
		if pending.index < len(entries) && sameInputEntry(entries[pending.index], pending.entry) {
			entries[pending.index].Text = optimisticDisplayText(entries[pending.index].Text)
		}
	}
	return entries
}

func optimisticDisplayText(value string) string {
	const maxRunes = 4096
	cut := len(value)
	count := 0
	for index := range value {
		if count == maxRunes {
			cut = index
			break
		}
		count++
	}
	if cut == len(value) {
		return value
	}
	remaining := utf8.RuneCountInString(value[cut:])
	return value[:cut] + fmt.Sprintf("\n\n… (%d more characters submitted)", remaining)
}

func (m *Model) updateViewport(message tea.Msg) tea.Cmd {
	var command tea.Cmd
	m.viewport, command = m.viewport.Update(message)
	m.followOutput = m.viewport.AtBottom()
	return command
}

func (m *Model) View() tea.View {
	parts := []string{m.viewport.View()}
	composerVisible := false
	if m.resolvingApprovalID != "" {
		parts = append(parts, render.ApprovalResolving(m.config.NoColor, m.width))
	} else if m.snapshot.Suspension != nil {
		parts = append(parts, render.Approval(m.snapshot.Suspension, m.approvalIndex, m.decisions, m.config.NoColor, m.width))
	} else if m.help {
		parts = append(parts, "Commands: /new /resume ID /help /clear /export /thinking /tools /quit\nKeys: enter submit, shift+enter/ctrl+j newline, alt+s steer, alt+f follow-up, alt+n next turn, ctrl+t thinking, ctrl+g tools, ctrl+e editor, ctrl+c interrupt")
	} else {
		composerVisible = true
		parts = append(parts, m.composer.View())
	}
	if banner := render.Banner(m.banner, m.failure, m.config.NoColor, m.width); banner != "" {
		parts = append(parts, banner)
	}
	footerState := m.snapshot.State
	if m.resolvingApprovalID != "" {
		footerState = uit.StateRunning
	}
	parts = append(parts,
		render.Status(m.snapshot, m.inflight > 0, m.config.NoColor, m.width),
		render.Footer(footerState, m.config.NoColor, m.width),
	)
	view := tea.NewView(strings.Join(parts, "\n"))
	view.AltScreen = ResolveAltScreen(m.config.AlternateScreen, m.environ)
	view.MouseMode = tea.MouseModeCellMotion
	view.WindowTitle = "Agentic Harness — " + m.snapshot.SessionID
	if m.config.NoColor {
		view.WindowTitle = "Agentic Harness - " + m.snapshot.SessionID
		if composerVisible {
			view.Cursor = m.composer.Cursor()
			if view.Cursor != nil {
				view.Cursor.Y += m.viewport.Height() + 1
			}
		}
	}
	return view
}

func (m *Model) LastExport() string { return m.lastExport }

// SessionID returns the currently attached durable session identity. It lets
// launchers persist a resume pointer without learning a host's storage layout.
func (m *Model) SessionID() string {
	if m == nil {
		return ""
	}
	if m.session != nil {
		return m.session.ID()
	}
	return m.snapshot.SessionID
}

func (m *Model) cycleThinking() {
	switch m.config.Thinking {
	case render.ThinkingVisible:
		m.config.Thinking = render.ThinkingCollapsed
	case render.ThinkingCollapsed:
		m.config.Thinking = render.ThinkingHidden
	default:
		m.config.Thinking = render.ThinkingVisible
	}
	m.banner, m.failure = "Thinking view: "+string(m.config.Thinking)+".", false
	m.syncViewport()
}

func ResolveAltScreen(mode AlternateScreen, environ []string) bool {
	if mode == AlternateAlways {
		return true
	}
	if mode == AlternateNever {
		return false
	}
	values := make(map[string]string, len(environ))
	for _, value := range environ {
		key, item, found := strings.Cut(value, "=")
		if found {
			values[key] = item
		}
	}
	return values["TMUX"] == "" && values["STY"] == "" && values["TERM"] != "dumb"
}

func upsertTool(values []uit.Tool, tool uit.Tool) []uit.Tool {
	for index := range values {
		if values[index].CallID == tool.CallID && tool.CallID != "" {
			missingSummary := tool.Summary == ""
			if tool.Name == "" {
				tool.Name = values[index].Name
			}
			if missingSummary {
				tool.Summary = values[index].Summary
			}
			if tool.Presentation == (uit.ToolPresentation{}) ||
				(missingSummary && values[index].Summary != "") {
				tool.Presentation = values[index].Presentation
			}
			values[index] = tool
			return values
		}
	}
	return append(values, tool)
}

func removeQueue(values []uit.QueuedInput, id string) []uit.QueuedInput {
	result := values[:0]
	for _, value := range values {
		if value.ID != id {
			result = append(result, value)
		}
	}
	return result
}
