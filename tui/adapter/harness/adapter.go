// Package harness adapts Agentic Harness sessions to the reusable TUI port.
package harness

import (
	"context"
	"errors"
	"fmt"
	"sync"

	agentic "github.com/regularkevvv/agentic"
	harnesscore "github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/observe"
	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"

	uit "github.com/regularkevvv/agentic/tui"
)

type config struct {
	profileLabel  string
	workspace     string
	execution     string
	toolPresenter uit.ToolPresenter
}

type Option func(*config)

func WithProfileLabel(value string) Option { return func(c *config) { c.profileLabel = value } }
func WithWorkspace(value string) Option    { return func(c *config) { c.workspace = value } }
func WithExecutionLabel(value string) Option {
	return func(c *config) { c.execution = value }
}
func WithToolPresenter(value uit.ToolPresenter) Option {
	return func(c *config) { c.toolPresenter = value }
}

type host[O any] struct {
	runtime *harnesscore.Harness[O]
	config  config
}

// New attaches to an already assembled Harness. It never constructs a model,
// provider, permission policy, capability graph, or code-mode executor.
func New[O any](runtime *harnesscore.Harness[O], options ...Option) (uit.Host, error) {
	if runtime == nil {
		return nil, errors.New("TUI Harness adapter requires a runtime")
	}
	value := config{}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("TUI Harness adapter option must not be nil")
		}
		option(&value)
	}
	return &host[O]{runtime: runtime, config: value}, nil
}

func (h *host[O]) NewSession(ctx context.Context, _ uit.SessionOptions) (uit.Session, error) {
	created, err := h.runtime.NewSession(ctx)
	if err != nil {
		return nil, err
	}
	return &session[O]{session: created, config: h.config}, nil
}

func (h *host[O]) ResumeSession(ctx context.Context, id string) (uit.Session, error) {
	if id == "" {
		return nil, errors.New("session ID is empty")
	}
	resumed, err := h.runtime.ResumeSession(ctx, id)
	if err != nil {
		return nil, err
	}
	return &session[O]{session: resumed, config: h.config}, nil
}

type session[O any] struct {
	session *harnesscore.Session[O]
	config  config
}

func (s *session[O]) ID() string { return s.session.ID() }

func (s *session[O]) Snapshot(ctx context.Context) (uit.Snapshot, error) {
	value, err := s.session.Snapshot(ctx)
	if err != nil {
		return uit.Snapshot{}, err
	}
	result := uit.Snapshot{
		SessionID: s.ID(), Cursor: value.Cursor, State: state(value.State),
		Transcript: messagesWithSummary(value.Messages, s.config.toolPresenter, s.session.ToolSummary),
		Usage:      usage(value.Usage), ProfileLabel: s.config.profileLabel, Workspace: s.config.workspace,
		Execution: s.config.execution,
	}
	result.Pending = make([]uit.QueuedInput, len(value.Pending))
	for index, pending := range value.Pending {
		result.Pending[index] = uit.QueuedInput{
			ID: pending.ID, Kind: uit.QueueKind(pending.Kind), Text: pending.Message.GetTextContent(),
		}
	}
	if value.Suspension != nil {
		result.Suspension, err = suspension(*value.Suspension)
		if err != nil {
			return uit.Snapshot{}, err
		}
	}
	return result, nil
}

func (s *session[O]) Subscribe(options uit.SubscribeOptions) uit.Subscription {
	source := s.session.Observe(observe.SubscribeOptions{
		AfterCursor: options.AfterCursor, Buffer: options.Buffer, Preview: options.Preview,
	})
	return mapSubscription(source, s, s.config.toolPresenter)
}

func (s *session[O]) Submit(ctx context.Context, input uit.Input) error {
	if err := input.Validate(); err != nil {
		return err
	}
	_, err := s.session.Prompt(ctx, agentic.NewTextMessage(agentic.RoleUser, input.Text))
	return err
}

func (s *session[O]) Steer(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, input, s.session.Steer)
}

func (s *session[O]) FollowUp(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, input, s.session.FollowUp)
}

func (s *session[O]) NextTurn(ctx context.Context, input uit.Input) error {
	return s.queue(ctx, input, s.session.NextTurn)
}

func (s *session[O]) queue(
	ctx context.Context,
	input uit.Input,
	accept func(context.Context, agentic.Message) (harnesscore.QueueReceipt, error),
) error {
	if err := input.Validate(); err != nil {
		return err
	}
	_, err := accept(ctx, agentic.NewTextMessage(agentic.RoleUser, input.Text))
	return err
}

func (s *session[O]) Resolve(ctx context.Context, resolution uit.Resolution) error {
	if err := resolution.Validate(); err != nil {
		return err
	}
	snapshot, err := s.session.Snapshot(ctx)
	if err != nil {
		return err
	}
	if snapshot.Suspension == nil || snapshot.Suspension.ID != resolution.SuspensionID {
		return errors.New("resolution does not match the current suspension")
	}
	approvals, err := permission.InspectSuspension(*snapshot.Suspension)
	if err != nil {
		return err
	}
	required := make(map[string]bool, len(approvals))
	for _, approval := range approvals {
		required[approval.CallID] = true
	}
	decisions := make(map[string]uit.Decision, len(resolution.Decisions))
	for _, decision := range resolution.Decisions {
		if !required[decision.CallID] {
			return fmt.Errorf("resolution contains unknown approval %q", decision.CallID)
		}
		decisions[decision.CallID] = decision
	}
	if len(decisions) != len(required) {
		return fmt.Errorf("resolution has %d decisions for %d approvals", len(decisions), len(required))
	}
	request := harnesscore.ResumeRequest{SuspensionID: resolution.SuspensionID}
	for _, approval := range approvals {
		decision := decisions[approval.CallID]
		action := harnesscore.ResolutionApprove
		if decision.Action == uit.DecisionDeny {
			action = harnesscore.ResolutionDeny
		}
		request.Resolutions = append(request.Resolutions, harnesscore.ToolResolution{
			CallID: approval.CallID, Action: action, Reason: decision.Reason,
		})
	}
	if resolution.Prompt != nil {
		prompt := agentic.NewTextMessage(agentic.RoleUser, resolution.Prompt.Text)
		request.Prompt = &prompt
	}
	_, err = s.session.Resume(ctx, request)
	return err
}

func (s *session[O]) Interrupt(ctx context.Context) error { return s.session.Interrupt(ctx) }
func (s *session[O]) Close(ctx context.Context) error     { return s.session.Close(ctx) }

func state(value harnesscore.SessionState) uit.State { return uit.State(value.String()) }

func usage(value agentic.Usage) uit.Usage {
	return uit.Usage{
		PromptTokens: value.PromptTokens, CompletionTokens: value.CompletionTokens,
		TotalTokens: value.TotalTokens, CacheReadTokens: value.CacheReadTokens,
		CacheCreationTokens: value.CacheCreationTokens, ReasoningTokens: value.ReasoningTokens,
		Requests: value.Requests, ToolCalls: value.ToolCalls,
	}
}

func suspension(value agentic.Suspension) (*uit.Suspension, error) {
	result := &uit.Suspension{ID: value.ID, Kind: value.Kind}
	approvals, err := permission.InspectSuspension(value)
	if err != nil {
		if errors.Is(err, harnessruntime.ErrUnsupportedDeferral) {
			result.Description = "This suspension has no registered terminal presenter; use the owning application to resolve it."
			return result, nil
		}
		return nil, fmt.Errorf("inspect permission suspension: %w", err)
	}
	result.Supported = true
	result.Approvals = make([]uit.Approval, len(approvals))
	for index, approval := range approvals {
		resource := approval.Request.CanonicalResource
		result.Approvals[index] = uit.Approval{
			CallID: approval.CallID, ToolName: approval.ToolName,
			Capability: approval.Request.Capability, Action: approval.Request.Action,
			ResourceScheme: resource.Scheme, CanonicalResource: resource.ID, ResourceDisplay: resource.Display,
		}
	}
	return result, nil
}

func messages(values []agentic.Message, presenters ...uit.ToolPresenter) []uit.Entry {
	return messagesWithSummary(values, firstPresenter(presenters), nil)
}

func messagesWithSummary(
	values []agentic.Message,
	presenter uit.ToolPresenter,
	summarize func(agentic.ToolUse) string,
) []uit.Entry {
	result := make([]uit.Entry, 0, len(values))
	for _, value := range values {
		entry := messageWithSummary(value, presenter, summarize)
		if entry.Role == uit.RoleSystem {
			continue
		}
		result = append(result, entry)
	}
	return result
}

func messageWithSummary(
	value agentic.Message,
	presenter uit.ToolPresenter,
	summarize func(agentic.ToolUse) string,
) uit.Entry {
	result := uit.Entry{Role: uit.Role(value.Role)}
	for _, part := range value.Content {
		switch part.Type {
		case agentic.ContentText:
			result.Text += part.Text
		case agentic.ContentThinking:
			if part.Thinking != nil {
				result.Thinking = append(result.Thinking, uit.Thinking{
					Text: part.Thinking.Text, ProviderName: part.Thinking.ProviderName,
					ThinkingID: part.Thinking.ID, Redacted: part.Thinking.IsRedacted(),
				})
			}
		case agentic.ContentToolUse:
			if part.ToolUse != nil {
				summary := ""
				if summarize != nil {
					summary = summarize(*part.ToolUse)
				}
				result.Tools = append(result.Tools, presentTool(uit.Tool{
					CallID: part.ToolUse.ID, Name: part.ToolUse.Name,
					State: uit.ToolPlanned, Summary: summary,
				}, presenter))
			}
		case agentic.ContentToolResult:
			if part.ToolResult != nil {
				toolState := uit.ToolDone
				if part.ToolResult.IsError {
					toolState = uit.ToolError
				}
				result.Tools = append(result.Tools, presentTool(uit.Tool{CallID: part.ToolResult.ToolUseID, Name: part.ToolResult.Name, State: toolState}, presenter))
			}
		}
	}
	return result
}

func mapObservation(value observe.Event, owner interface {
	Snapshot(context.Context) (uit.Snapshot, error)
}, presenters ...uit.ToolPresenter) (uit.Event, error) {
	presenter := firstPresenter(presenters)
	result := uit.Event{
		Cursor: value.Cursor, Ordinal: value.Ordinal, Durable: value.Nature != agentic.EventPreview,
		SessionID: value.SessionID, ParentID: value.ParentID, Agent: value.Agent,
		Depth: value.Depth, Turn: value.Turn, Kind: uit.EventKind(value.Kind),
		TextDelta: value.TextDelta, State: uit.State(value.State), Dropped: value.Dropped,
	}
	if value.Message != nil {
		entry := observedMessage(*value.Message, presenter)
		result.Entry = &entry
	}
	for _, current := range value.Messages {
		result.Entries = append(result.Entries, observedMessage(current, presenter))
	}
	if value.Thinking != nil {
		thinking := uit.Thinking{Text: value.Thinking.Text, ProviderName: value.Thinking.ProviderName, ThinkingID: value.Thinking.ThinkingID, Redacted: value.Thinking.Redacted}
		result.Thinking = &thinking
	}
	if value.Tool != nil {
		tool := observedTool(*value.Tool, presenter)
		result.Tool = &tool
	}
	for _, current := range value.Tools {
		result.Tools = append(result.Tools, observedTool(current, presenter))
	}
	if value.Usage != nil {
		mapped := uit.Usage{
			PromptTokens: value.Usage.PromptTokens, CompletionTokens: value.Usage.CompletionTokens,
			TotalTokens: value.Usage.TotalTokens, CacheReadTokens: value.Usage.CacheReadTokens,
			CacheCreationTokens: value.Usage.CacheCreationTokens, ReasoningTokens: value.Usage.ReasoningTokens,
			Requests: value.Usage.Requests, ToolCalls: value.Usage.ToolCalls,
		}
		result.Usage = &mapped
	}
	if value.Suspension != nil {
		// The full approval view is reconciled from Snapshot, never reconstructed
		// from a partial event payload.
		snapshot, err := owner.Snapshot(context.Background())
		if err != nil {
			return uit.Event{}, err
		}
		result.Suspension = snapshot.Suspension
	}
	if value.Failure != nil {
		result.Failure = value.Failure.Message
	}
	if value.Queue != nil {
		queued := uit.QueuedInput{ID: value.Queue.ID, Kind: uit.QueueKind(value.Queue.Kind)}
		if value.Queue.Message != nil {
			queued.Text = value.Queue.Message.Text
		}
		result.Queue = &queued
	}
	return result, nil
}

func observedMessage(value observe.Message, presenters ...uit.ToolPresenter) uit.Entry {
	result := uit.Entry{Role: uit.Role(value.Role), Text: value.Text}
	presenter := firstPresenter(presenters)
	for _, current := range value.Thinking {
		result.Thinking = append(result.Thinking, uit.Thinking{
			Text: current.Text, ProviderName: current.ProviderName, ThinkingID: current.ThinkingID, Redacted: current.Redacted,
		})
	}
	for _, current := range value.Tools {
		result.Tools = append(result.Tools, observedTool(current, presenter))
	}
	return result
}

func observedTool(value observe.Tool, presenters ...uit.ToolPresenter) uit.Tool {
	tool := uit.Tool{CallID: value.CallID, Name: value.Name, State: uit.ToolState(value.State), Attempt: value.Attempt, Summary: value.Summary}
	return presentTool(tool, firstPresenter(presenters))
}

func presentTool(tool uit.Tool, presenter uit.ToolPresenter) uit.Tool {
	if presenter != nil {
		tool.Presentation = presenter.PresentTool(tool)
	}
	return tool
}

func firstPresenter(values []uit.ToolPresenter) uit.ToolPresenter {
	if len(values) == 0 {
		return nil
	}
	return values[0]
}

type mappedSubscription struct {
	events <-chan uit.Event
	errors <-chan error
	close  func()
	once   sync.Once
}

func (s *mappedSubscription) Events() <-chan uit.Event { return s.events }
func (s *mappedSubscription) Errors() <-chan error     { return s.errors }
func (s *mappedSubscription) Close() {
	if s == nil {
		return
	}
	s.once.Do(s.close)
}

func mapSubscription(source observe.Subscription, owner interface {
	Snapshot(context.Context) (uit.Snapshot, error)
}, presenters ...uit.ToolPresenter) uit.Subscription {
	presenter := firstPresenter(presenters)
	events := make(chan uit.Event)
	errors := make(chan error, 1)
	done := make(chan struct{})
	var once sync.Once
	closeFn := func() {
		once.Do(func() {
			close(done)
			if source != nil {
				source.Close()
			}
		})
	}
	result := &mappedSubscription{events: events, errors: errors, close: closeFn}
	go func() {
		defer close(events)
		defer close(errors)
		defer closeFn()
		if source == nil {
			return
		}
		observations, sourceErrors := source.Events(), source.Errors()
		for observations != nil || sourceErrors != nil {
			select {
			case <-done:
				return
			case value, ok := <-observations:
				if !ok {
					observations = nil
					continue
				}
				mapped, err := mapObservation(value, owner, presenter)
				if err != nil {
					errors <- err
					return
				}
				select {
				case <-done:
					return
				case events <- mapped:
				}
			case err, ok := <-sourceErrors:
				if !ok {
					sourceErrors = nil
					continue
				}
				if err != nil {
					errors <- err
				}
				return
			}
		}
	}()
	return result
}

var _ uit.Host = (*host[string])(nil)
var _ uit.Session = (*session[string])(nil)
