// Package contextpolicy builds provider views from an append-only durable
// transcript. It never rewrites transcript truth.
package contextpolicy

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	agentic "github.com/regularkevvv/agentic"
)

var (
	ErrInvalidConfig         = errors.New("invalid context policy configuration")
	ErrInvalidTransform      = errors.New("context transform violated append-only rules")
	ErrCompactionInvalid     = errors.New("context compaction does not match the durable prefix")
	ErrContextWindowExceeded = errors.New("context cannot be compacted to the configured target")
)

// Transcript is the append-only durable input visible to context transforms.
type Transcript struct {
	Messages []agentic.Message
}

// TransformContext separates durable additions from ephemeral tail content.
// Durable mutations must be append-only. Ephemeral messages are re-derived for
// one provider request and are never persisted by the policy itself.
type TransformContext struct {
	Durable   *Transcript
	Ephemeral *[]agentic.Message
}

// Transform contributes ordered context policy.
type Transform interface {
	Transform(context.Context, *TransformContext) error
}

// TransformFunc adapts a function to Transform.
type TransformFunc func(context.Context, *TransformContext) error

func (f TransformFunc) Transform(ctx context.Context, value *TransformContext) error {
	return f(ctx, value)
}

// TokenCounter counts model-specific tokens for one canonical byte segment.
type TokenCounter interface {
	Count(context.Context, []byte) (int, error)
}

// TokenCounterFunc adapts a function to TokenCounter.
type TokenCounterFunc func(context.Context, []byte) (int, error)

func (f TokenCounterFunc) Count(ctx context.Context, value []byte) (int, error) {
	return f(ctx, value)
}

// ByteCounter conservatively treats every UTF-8 byte as one token.
type ByteCounter struct{}

func (ByteCounter) Count(_ context.Context, value []byte) (int, error) {
	return len(value), nil
}

// Config controls cache geometry and compaction thresholds.
type Config struct {
	ContextWindowTokens int
	TriggerPercent      int
	TargetPercent       int
	RecentMessages      int
	MessageOverhead     int
	PartOverhead        int
	ToolOverhead        int
	Counter             TokenCounter
	Tools               []agentic.Tool
}

func (c Config) normalized(compactor Compactor) (Config, error) {
	if c.Counter == nil {
		c.Counter = ByteCounter{}
	}
	if c.TriggerPercent == 0 {
		c.TriggerPercent = 70
	}
	if c.TargetPercent == 0 {
		c.TargetPercent = 50
	}
	if c.RecentMessages == 0 {
		c.RecentMessages = 8
	}
	if c.MessageOverhead == 0 {
		c.MessageOverhead = 8
	}
	if c.PartOverhead == 0 {
		c.PartOverhead = 4
	}
	if c.ToolOverhead == 0 {
		c.ToolOverhead = 12
	}
	if c.ContextWindowTokens < 0 || c.TriggerPercent < 1 || c.TriggerPercent > 100 ||
		c.TargetPercent < 1 || c.TargetPercent >= c.TriggerPercent ||
		c.RecentMessages < 0 || c.MessageOverhead < 0 || c.PartOverhead < 0 ||
		c.ToolOverhead < 0 {
		return Config{}, ErrInvalidConfig
	}
	if compactor != nil && c.ContextWindowTokens <= 0 {
		return Config{}, fmt.Errorf("%w: a compactor requires a positive context window", ErrInvalidConfig)
	}
	tools, err := cloneTools(c.Tools)
	if err != nil {
		return Config{}, err
	}
	c.Tools = tools
	return c, nil
}

// Compaction is a durable recipe for replacing one stable prefix in a provider
// projection. Start leaves an initial system message outside the replacement.
type Compaction struct {
	Version    int             `json:"version"`
	Start      int             `json:"start"`
	Cut        int             `json:"cut"`
	PrefixHash string          `json:"prefix_hash"`
	Summary    agentic.Message `json:"summary"`
}

// ProjectionRequest carries the current derived durable view and the most
// recently persisted compaction recipe.
type ProjectionRequest struct {
	Messages   []agentic.Message
	Compaction *Compaction
}

// Projection is a one-request provider view plus facts the session must append
// durably before performing the model request.
type Projection struct {
	Messages          []agentic.Message
	DurableAdditions  []agentic.Message
	Compaction        *Compaction
	CompactionChanged bool
}

// Projector constructs a provider view without mutating its input.
type Projector interface {
	Project(context.Context, ProjectionRequest) (Projection, error)
}

// ProjectorFunc adapts a function to Projector.
type ProjectorFunc func(context.Context, ProjectionRequest) (Projection, error)

func (f ProjectorFunc) Project(ctx context.Context, request ProjectionRequest) (Projection, error) {
	return f(ctx, request)
}

// Compactor summarizes a protocol-valid prefix chosen by Policy.
type Compactor interface {
	Summarize(context.Context, []agentic.Message) (agentic.Message, error)
}

// CompactorFunc adapts a function to Compactor.
type CompactorFunc func(context.Context, []agentic.Message) (agentic.Message, error)

func (f CompactorFunc) Summarize(ctx context.Context, messages []agentic.Message) (agentic.Message, error) {
	return f(ctx, messages)
}

// Policy applies ordered transforms, cache-aware compaction, and no transcript
// mutation. Transcript repair is intentionally installed by session after this
// projector so Repair remains terminal.
type Policy struct {
	config     Config
	transforms []Transform
	compactor  Compactor
}

func New(config Config, transforms []Transform, compactor Compactor) (*Policy, error) {
	normalized, err := config.normalized(compactor)
	if err != nil {
		return nil, err
	}
	copied := append([]Transform(nil), transforms...)
	for _, transform := range copied {
		if transform == nil {
			return nil, fmt.Errorf("%w: nil transform", ErrInvalidConfig)
		}
	}
	return &Policy{config: normalized, transforms: copied, compactor: compactor}, nil
}

// Passthrough returns an immutable projector with no transforms or compaction.
func Passthrough() Projector {
	policy, _ := New(Config{}, nil, nil)
	return policy
}

func (p *Policy) Project(ctx context.Context, request ProjectionRequest) (Projection, error) {
	if err := ctx.Err(); err != nil {
		return Projection{}, err
	}
	original := cloneMessages(request.Messages)
	durable := Transcript{Messages: cloneMessages(request.Messages)}
	ephemeral := []agentic.Message(nil)
	transformContext := &TransformContext{Durable: &durable, Ephemeral: &ephemeral}
	for _, transform := range p.transforms {
		beforeDurable := cloneMessages(transformContext.Durable.Messages)
		if err := transform.Transform(ctx, transformContext); err != nil {
			return Projection{}, err
		}
		if transformContext.Durable == nil || transformContext.Ephemeral == nil {
			return Projection{}, fmt.Errorf("%w: transform cleared a context partition", ErrInvalidTransform)
		}
		if len(transformContext.Durable.Messages) < len(beforeDurable) ||
			!messagesEqual(transformContext.Durable.Messages[:len(beforeDurable)], beforeDurable) {
			return Projection{}, fmt.Errorf("%w: durable transcript was rewritten", ErrInvalidTransform)
		}
		if err := validateContextTail(transformContext.Durable.Messages[len(beforeDurable):]); err != nil {
			return Projection{}, err
		}
		if err := validateContextTail(*transformContext.Ephemeral); err != nil {
			return Projection{}, err
		}
	}
	additions := cloneMessages(transformContext.Durable.Messages[len(original):])
	view := append(
		cloneMessages(transformContext.Durable.Messages),
		cloneMessages(*transformContext.Ephemeral)...,
	)
	current := cloneCompaction(request.Compaction)
	if current != nil {
		compacted, err := Apply(view, *current)
		if err != nil {
			return Projection{}, err
		}
		view = compacted
	}
	if p.compactor == nil || p.config.ContextWindowTokens == 0 {
		return Projection{
			Messages:         view,
			DurableAdditions: additions,
			Compaction:       current,
		}, nil
	}
	estimate, err := p.Estimate(ctx, view)
	if err != nil {
		return Projection{}, err
	}
	trigger := p.config.ContextWindowTokens * p.config.TriggerPercent / 100
	if estimate < trigger {
		return Projection{
			Messages:         view,
			DurableAdditions: additions,
			Compaction:       current,
		}, nil
	}

	start, cut, ok, err := compactionCut(transformContext.Durable.Messages, p.config.RecentMessages)
	if err != nil {
		return Projection{}, err
	}
	if !ok {
		return Projection{}, ErrContextWindowExceeded
	}
	summary, err := p.compactor.Summarize(ctx, cloneMessages(transformContext.Durable.Messages[start:cut]))
	if err != nil {
		return Projection{}, err
	}
	if err := validateContextTail([]agentic.Message{summary}); err != nil {
		return Projection{}, fmt.Errorf("%w: compactor summary: %v", ErrCompactionInvalid, err)
	}
	next := &Compaction{
		Version:    1,
		Start:      start,
		Cut:        cut,
		PrefixHash: prefixHash(transformContext.Durable.Messages[start:cut]),
		Summary:    cloneMessages([]agentic.Message{summary})[0],
	}
	full := append(
		cloneMessages(transformContext.Durable.Messages),
		cloneMessages(*transformContext.Ephemeral)...,
	)
	compacted, err := Apply(full, *next)
	if err != nil {
		return Projection{}, err
	}
	estimate, err = p.Estimate(ctx, compacted)
	if err != nil {
		return Projection{}, err
	}
	target := p.config.ContextWindowTokens * p.config.TargetPercent / 100
	if estimate > target {
		return Projection{}, fmt.Errorf("%w: estimate %d exceeds target %d", ErrContextWindowExceeded, estimate, target)
	}
	return Projection{
		Messages:          compacted,
		DurableAdditions:  additions,
		Compaction:        next,
		CompactionChanged: !compactionsEqual(current, next),
	}, nil
}

// Estimate counts messages, content framing, and every configured tool schema.
func (p *Policy) Estimate(ctx context.Context, messages []agentic.Message) (int, error) {
	total := 0
	for _, message := range messages {
		encoded, err := json.Marshal(message)
		if err != nil {
			return 0, fmt.Errorf("encode context message: %w", err)
		}
		count, err := p.config.Counter.Count(ctx, encoded)
		if err != nil {
			return 0, fmt.Errorf("count context message: %w", err)
		}
		total += count + p.config.MessageOverhead + len(message.Content)*p.config.PartOverhead
	}
	for _, tool := range p.config.Tools {
		encoded, err := json.Marshal(tool)
		if err != nil {
			return 0, fmt.Errorf("encode tool schema: %w", err)
		}
		count, err := p.config.Counter.Count(ctx, encoded)
		if err != nil {
			return 0, fmt.Errorf("count tool schema: %w", err)
		}
		total += count + p.config.ToolOverhead
	}
	return total, nil
}

// Apply reapplies a persisted compaction to a later transcript whose compacted
// prefix is unchanged.
func Apply(messages []agentic.Message, compaction Compaction) ([]agentic.Message, error) {
	if compaction.Version != 1 || compaction.Start < 0 || compaction.Cut <= compaction.Start ||
		compaction.Cut > len(messages) || compaction.Summary.Role != agentic.RoleUser {
		return nil, ErrCompactionInvalid
	}
	if prefixHash(messages[compaction.Start:compaction.Cut]) != compaction.PrefixHash {
		return nil, ErrCompactionInvalid
	}
	result := make([]agentic.Message, 0, len(messages)-(compaction.Cut-compaction.Start)+1)
	result = append(result, cloneMessages(messages[:compaction.Start])...)
	result = append(result, cloneMessages([]agentic.Message{compaction.Summary})...)
	result = append(result, cloneMessages(messages[compaction.Cut:])...)
	return result, nil
}

func validateContextTail(messages []agentic.Message) error {
	for _, message := range messages {
		if message.Role != agentic.RoleUser {
			return fmt.Errorf("%w: context additions must use the user role", ErrInvalidTransform)
		}
		if len(message.Content) == 0 {
			return fmt.Errorf("%w: context additions must not be empty", ErrInvalidTransform)
		}
		for _, part := range message.Content {
			if part.Type != agentic.ContentText {
				return fmt.Errorf("%w: context additions must contain text only", ErrInvalidTransform)
			}
		}
	}
	return nil
}

func cloneMessages(messages []agentic.Message) []agentic.Message {
	if len(messages) == 0 {
		return nil
	}
	encoded, err := json.Marshal(messages)
	if err == nil {
		var result []agentic.Message
		if json.Unmarshal(encoded, &result) == nil {
			return result
		}
	}
	return append([]agentic.Message(nil), messages...)
}

func cloneTools(tools []agentic.Tool) ([]agentic.Tool, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	encoded, err := json.Marshal(tools)
	if err != nil {
		return nil, fmt.Errorf("clone tool schemas: %w", err)
	}
	var result []agentic.Tool
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, fmt.Errorf("clone tool schemas: %w", err)
	}
	return result, nil
}

func cloneCompaction(value *Compaction) *Compaction {
	if value == nil {
		return nil
	}
	copy := *value
	copy.Summary = cloneMessages([]agentic.Message{value.Summary})[0]
	return &copy
}

func compactionsEqual(left, right *Compaction) bool {
	return reflect.DeepEqual(left, right)
}

func messagesEqual(left, right []agentic.Message) bool {
	if len(left) != len(right) {
		return false
	}
	if len(left) == 0 {
		return true
	}
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func prefixHash(messages []agentic.Message) string {
	encoded, _ := json.Marshal(struct {
		Version  int
		Messages []agentic.Message
	}{Version: 1, Messages: messages})
	digest := sha256.Sum256(encoded)
	return "ctx1:" + hex.EncodeToString(digest[:])
}

func compactionCut(messages []agentic.Message, recent int) (int, int, bool, error) {
	start := 0
	if len(messages) > 0 && messages[0].Role == agentic.RoleSystem {
		start = 1
	}
	maxCut := len(messages) - recent
	if maxCut <= start {
		return 0, 0, false, nil
	}
	open := make(map[string]bool)
	best := -1
	for index, message := range messages {
		uses := message.GetToolUses()
		if len(uses) > 0 {
			if len(open) != 0 {
				return 0, 0, false, fmt.Errorf("%w: overlapping tool frontiers", ErrCompactionInvalid)
			}
			for _, call := range uses {
				if call.ID == "" || open[call.ID] {
					return 0, 0, false, fmt.Errorf("%w: duplicate tool call ID", ErrCompactionInvalid)
				}
				open[call.ID] = true
			}
		}
		for _, result := range message.GetToolResults() {
			if !open[result.ToolUseID] {
				return 0, 0, false, fmt.Errorf("%w: orphan tool result", ErrCompactionInvalid)
			}
			delete(open, result.ToolUseID)
		}
		boundary := index + 1
		if boundary > start && boundary <= maxCut && len(open) == 0 {
			best = boundary
		}
	}
	if len(open) != 0 {
		// An active deferred frontier stays in the recent reserve; it must never
		// be split or sent through a compactor.
		if best < 0 {
			return 0, 0, false, nil
		}
	}
	if best < 0 {
		return 0, 0, false, nil
	}
	return start, best, true, nil
}
