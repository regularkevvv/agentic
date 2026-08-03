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

	composer      textarea.Model
	viewport      viewport.Model
	width         int
	height        int
	preview       string
	thinking      string
	liveTools     []uit.Tool
	clearBefore   int
	inflight      int
	banner        string
	failure       bool
	help          bool
	closing       bool
	approvalIndex int
	decisions     map[string]uit.DecisionAction
	history       []string
	historyIndex  int
	lastExport    string
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
	composer.SetHeight(3)
	composer.SetWidth(80)
	composer.Focus()
	vp := viewport.New(viewport.WithWidth(80), viewport.WithHeight(16))
	vp.SoftWrap = true
	return &Model{
		host: host, ctx: ctx, cancel: cancel, config: config, resumeID: options.ResumeID,
		editor: options.Editor, terminalEditor: options.TerminalEditor, environ: environ, composer: composer, viewport: vp,
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
	snapshot *uit.Snapshot
	message  string
	export   string
	err      error
	quit     bool
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
		if msg.err != nil {
			m.setError(msg.err)
		} else {
			m.banner, m.failure = msg.message, false
			if msg.snapshot != nil {
				m.snapshot = *msg.snapshot
				m.preview, m.thinking, m.liveTools = "", "", nil
				m.resetApproval()
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
	m.composer, command = m.composer.Update(message)
	commands = append(commands, command)
	m.viewport, command = m.viewport.Update(message)
	commands = append(commands, command)
	return m, tea.Batch(commands...)
}

func (m *Model) handleKey(key tea.KeyPressMsg) (tea.Cmd, bool) {
	stroke := key.Keystroke()
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
	case "enter":
		return m.dispatchDraft("submit"), true
	case "alt+s":
		return m.dispatchDraft("steer"), true
	case "alt+f":
		return m.dispatchDraft("follow-up"), true
	case "alt+n":
		return m.dispatchDraft("next-turn"), true
	case "up":
		if len(m.history) > 0 && m.composer.Value() == "" {
			m.historyIndex = len(m.history) - 1
			m.composer.SetValue(m.history[m.historyIndex])
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
	}, action+" accepted")
}

func (m *Model) operationCmd(operation func(context.Context) error, success string) tea.Cmd {
	session := m.session
	return func() tea.Msg {
		if err := operation(m.ctx); err != nil {
			return operationDoneMsg{err: err}
		}
		snapshot, err := session.Snapshot(m.ctx)
		return operationDoneMsg{snapshot: &snapshot, message: success, err: err}
	}
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
		if oldBridge != nil {
			oldBridge.Close()
		}
		if old != nil {
			if err := old.Close(m.ctx); err != nil {
				return sessionReadyMsg{err: err}
			}
		}
		return openSessionCmd(m.ctx, m.host, id, m.config)()
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
		if bridge != nil {
			bridge.Close()
		}
		if session != nil {
			if err := session.Close(m.ctx); err != nil {
				return operationDoneMsg{err: err}
			}
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
		m.inflight++
		return m.operationCmd(func(ctx context.Context) error { return session.Resolve(ctx, resolution) }, "Suspension resolved.")
	case "esc":
		m.banner, m.failure = "Suspension left unresolved for safe handoff.", false
	}
	return nil
}

func (m *Model) applyBridge(update bridgeUpdate) {
	if update.err != nil {
		m.setError(update.err)
		return
	}
	if update.snapshot != nil {
		m.snapshot = *update.snapshot
		m.preview, m.thinking, m.liveTools = "", "", nil
		m.resetApproval()
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
		m.resetApproval()
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
		m.snapshot.Transcript = append(m.snapshot.Transcript, event.Entries...)
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
	case uit.EventRunCompleted, uit.EventRunEnded, uit.EventRunInterrupted:
		m.snapshot.State = uit.StateIdle
		m.preview, m.thinking, m.liveTools = "", "", nil
	case uit.EventRunFailed, uit.EventSessionFaulted:
		m.snapshot.State = uit.StateFaulted
	case uit.EventSessionClosed:
		m.snapshot.State = uit.StateClosed
	}
}

func (m *Model) replaceSession(session uit.Session, snapshot uit.Snapshot, bridge *eventBridge) {
	if m.bridge != nil && m.bridge != bridge {
		m.bridge.Close()
	}
	m.session, m.snapshot, m.bridge = session, snapshot, bridge
	m.clearBefore = 0
	m.preview, m.thinking, m.liveTools = "", "", nil
	m.banner, m.failure = "Session "+session.ID()+" ready.", false
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
	m.composer.SetWidth(max(10, m.width-2))
	composerHeight := min(5, max(3, m.height/5))
	m.composer.SetHeight(composerHeight)
	m.viewport.SetWidth(m.width)
	m.viewport.SetHeight(max(1, m.height-composerHeight-4))
	m.syncViewport()
}

func (m *Model) syncViewport() {
	entries := m.snapshot.Transcript
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
	m.viewport.GotoBottom()
}

func (m *Model) View() tea.View {
	parts := []string{m.viewport.View()}
	if m.snapshot.Suspension != nil {
		parts = append(parts, render.Approval(m.snapshot.Suspension, m.approvalIndex, m.decisions, m.config.NoColor))
	} else if m.help {
		parts = append(parts, "Commands: /new /resume ID /help /clear /export /thinking /tools /quit\nKeys: enter submit, alt+s steer, alt+f follow-up, alt+n next turn, ctrl+t thinking, ctrl+g tools, ctrl+e editor, ctrl+c interrupt")
	} else {
		parts = append(parts, m.composer.View())
	}
	if banner := render.Banner(m.banner, m.failure, m.config.NoColor); banner != "" {
		parts = append(parts, banner)
	}
	parts = append(parts, render.Status(m.snapshot, m.inflight > 0, m.config.NoColor), render.Footer(m.snapshot.State, m.config.NoColor))
	view := tea.NewView(strings.Join(parts, "\n"))
	view.AltScreen = ResolveAltScreen(m.config.AlternateScreen, m.environ)
	view.WindowTitle = "Agentic Harness — " + m.snapshot.SessionID
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
