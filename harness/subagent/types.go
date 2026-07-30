// Package subagent implements in-process, capture-restricted child agents as
// ordinary harness capabilities. Out-of-process agents and topology presets are
// intentionally outside this package.
package subagent

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/capability"
)

// DefaultSummaryBytes bounds the child summary embedded in a delegation tool
// result when Config.SummaryBytes is not set.
const DefaultSummaryBytes = 16 * 1024

var (
	ErrInvalidCapture = errors.New("invalid subagent capture policy")
	ErrDepthExceeded  = errors.New("subagent recursion depth exceeded")
	ErrChildNotFound  = errors.New("addressed child session is not running")
)

// Mode describes whether one parent resource is isolated, shared, or narrowed.
// ModeDefault resolves independently for each Capture field.
type Mode uint8

const (
	ModeDefault Mode = iota
	ModeIsolate
	ModeShare
	ModeNarrow
)

// Capture is resolved once when the capability is constructed. Defaults follow
// the production design: isolated history/tools, shared dependencies,
// environment, and budget, and non-broadening parent permissions.
type Capture struct {
	History      Mode
	Dependencies Mode
	Environment  Mode
	Tools        Mode
	Permissions  Mode
	Budget       Mode
}

// HistoryProjector produces the bounded or redacted transcript used by
// ModeNarrow history capture.
type HistoryProjector func(context.Context, []agentic.Message) ([]agentic.Message, error)

// Config defines one named delegation tool and the substrate used by every
// child session it creates.
type Config struct {
	Name        string
	Description string
	Ordering    capability.Ordering
	Capture     Capture

	Runtime      harness.RuntimeConfig
	Capabilities []harness.Capability

	HistoryProjector HistoryProjector
	EnvironmentRoot  string
	EnvironmentShell bool
	ToolFilter       func(string) bool
	Budget           *agentic.UsageLimits

	// MaxDepth includes the first child. Zero defaults to one, which permits
	// delegation from the parent but removes delegation tools from the child.
	MaxDepth     int
	SummaryBytes int
}

// Result is the bounded model-visible result of one child session. Full child
// messages remain in the child's separate durable journal.
type Result struct {
	Agent     string                  `json:"agent"`
	SessionID string                  `json:"session_id"`
	Status    agentic.ExecutionStatus `json:"status"`
	Summary   string                  `json:"summary"`
	FullBytes int                     `json:"full_bytes"`
	Truncated bool                    `json:"truncated"`
	Usage     agentic.Usage           `json:"usage"`
}

// Address is required for every external child inbox mutation. A parent
// Session.Steer call never consults this router.
type Address struct {
	ParentSessionID string `json:"parent_session_id"`
	ChildSessionID  string `json:"child_session_id"`
}

type childControl interface {
	Steer(context.Context, agentic.Message) (harness.QueueReceipt, error)
	FollowUp(context.Context, agentic.Message) (harness.QueueReceipt, error)
	NextTurn(context.Context, agentic.Message) (harness.QueueReceipt, error)
	Interrupt(context.Context) error
	Snapshot(context.Context) (harness.Snapshot, error)
}

type router struct {
	mu       sync.RWMutex
	children map[Address]childControl
}

func newRouter() *router {
	return &router{children: make(map[Address]childControl)}
}

func (r *router) add(address Address, child childControl) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.children[address]; exists {
		return fmt.Errorf("child route already exists: %+v", address)
	}
	r.children[address] = child
	return nil
}

func (r *router) remove(address Address, child childControl) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.children[address] == child {
		delete(r.children, address)
	}
}

func (r *router) child(address Address) (childControl, error) {
	if address.ParentSessionID == "" || address.ChildSessionID == "" {
		return nil, fmt.Errorf("%w: complete address is required", ErrChildNotFound)
	}
	r.mu.RLock()
	child := r.children[address]
	r.mu.RUnlock()
	if child == nil {
		return nil, fmt.Errorf("%w: parent=%s child=%s", ErrChildNotFound, address.ParentSessionID, address.ChildSessionID)
	}
	return child, nil
}

func (r *router) addresses(parentSessionID string) []Address {
	r.mu.RLock()
	result := make([]Address, 0, len(r.children))
	for address := range r.children {
		if parentSessionID == "" || address.ParentSessionID == parentSessionID {
			result = append(result, address)
		}
	}
	r.mu.RUnlock()
	sort.Slice(result, func(i, j int) bool {
		if result[i].ParentSessionID != result[j].ParentSessionID {
			return result[i].ParentSessionID < result[j].ParentSessionID
		}
		return result[i].ChildSessionID < result[j].ChildSessionID
	})
	return result
}

func resolveConfig(config Config) (Config, error) {
	config.Name = strings.TrimSpace(config.Name)
	config.Description = strings.TrimSpace(config.Description)
	if config.Name == "" || config.Description == "" {
		return Config{}, errors.New("subagent name and description are required")
	}
	if config.MaxDepth == 0 {
		config.MaxDepth = 1
	}
	if config.MaxDepth < 1 {
		return Config{}, errors.New("subagent maximum depth must be positive")
	}
	if config.SummaryBytes == 0 {
		config.SummaryBytes = DefaultSummaryBytes
	}
	if config.SummaryBytes < 1 {
		return Config{}, errors.New("subagent summary limit must be positive")
	}
	config.Capture = resolveCapture(config.Capture)
	if err := validateCapture(config); err != nil {
		return Config{}, err
	}
	config.Capabilities = append([]harness.Capability(nil), config.Capabilities...)
	config.Ordering.Before = append([]string(nil), config.Ordering.Before...)
	config.Ordering.After = append([]string(nil), config.Ordering.After...)
	if config.Budget != nil {
		copy := cloneLimits(*config.Budget)
		config.Budget = &copy
		if err := validateLimits(copy); err != nil {
			return Config{}, err
		}
	}
	return config, nil
}

func resolveCapture(capture Capture) Capture {
	if capture.History == ModeDefault {
		capture.History = ModeIsolate
	}
	if capture.Dependencies == ModeDefault {
		capture.Dependencies = ModeShare
	}
	if capture.Environment == ModeDefault {
		capture.Environment = ModeShare
	}
	if capture.Tools == ModeDefault {
		capture.Tools = ModeIsolate
	}
	if capture.Permissions == ModeDefault {
		capture.Permissions = ModeNarrow
	}
	if capture.Budget == ModeDefault {
		capture.Budget = ModeShare
	}
	return capture
}

func validateCapture(config Config) error {
	if !oneOf(config.Capture.History, ModeIsolate, ModeShare, ModeNarrow) {
		return fmt.Errorf("%w: history mode", ErrInvalidCapture)
	}
	if config.Capture.History == ModeNarrow && config.HistoryProjector == nil {
		return fmt.Errorf("%w: narrow history requires a projector", ErrInvalidCapture)
	}
	if !oneOf(config.Capture.Dependencies, ModeIsolate, ModeShare, ModeNarrow) {
		return fmt.Errorf("%w: dependencies mode", ErrInvalidCapture)
	}
	if !oneOf(config.Capture.Environment, ModeIsolate, ModeShare, ModeNarrow) {
		return fmt.Errorf("%w: environment mode", ErrInvalidCapture)
	}
	if config.Capture.Environment == ModeNarrow && strings.TrimSpace(config.EnvironmentRoot) == "" {
		return fmt.Errorf("%w: narrow environment requires a root", ErrInvalidCapture)
	}
	if !oneOf(config.Capture.Tools, ModeIsolate, ModeShare, ModeNarrow) {
		return fmt.Errorf("%w: tools mode", ErrInvalidCapture)
	}
	if config.Capture.Tools == ModeNarrow && config.ToolFilter == nil {
		return fmt.Errorf("%w: narrow tools require a filter", ErrInvalidCapture)
	}
	if !oneOf(config.Capture.Permissions, ModeIsolate, ModeShare, ModeNarrow) {
		return fmt.Errorf("%w: permissions mode", ErrInvalidCapture)
	}
	if !oneOf(config.Capture.Budget, ModeIsolate, ModeShare, ModeNarrow) {
		return fmt.Errorf("%w: budget mode", ErrInvalidCapture)
	}
	if config.Capture.Budget == ModeNarrow && config.Budget == nil {
		return fmt.Errorf("%w: narrow budget requires limits", ErrInvalidCapture)
	}
	if config.Capture.Budget == ModeShare && config.Budget != nil {
		return fmt.Errorf("%w: shared budget cannot add child-only limits; use narrow mode", ErrInvalidCapture)
	}
	if config.MaxDepth > 1 && config.Capture.Tools == ModeIsolate {
		return fmt.Errorf("%w: recursive delegation requires shared or narrowed tools", ErrInvalidCapture)
	}
	return nil
}

func oneOf(value Mode, allowed ...Mode) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}

func validateLimits(limits agentic.UsageLimits) error {
	for name, value := range map[string]*int{
		"request tokens":  limits.MaxRequestTokens,
		"response tokens": limits.MaxResponseTokens,
		"total tokens":    limits.MaxTotalTokens,
		"requests":        limits.MaxRequests,
		"tool calls":      limits.MaxToolCalls,
	} {
		if value != nil && *value < 1 {
			return fmt.Errorf("subagent budget %s must be positive", name)
		}
	}
	return nil
}

func cloneLimits(limits agentic.UsageLimits) agentic.UsageLimits {
	return agentic.UsageLimits{
		MaxRequestTokens:  cloneInt(limits.MaxRequestTokens),
		MaxResponseTokens: cloneInt(limits.MaxResponseTokens),
		MaxTotalTokens:    cloneInt(limits.MaxTotalTokens),
		MaxRequests:       cloneInt(limits.MaxRequests),
		MaxToolCalls:      cloneInt(limits.MaxToolCalls),
	}
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}
