package subagent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func runChild[O any](
	ctx context.Context,
	topology *Capability,
	task string,
	runner agentic.Runner[O],
) (Result, error) {
	task = strings.TrimSpace(task)
	if task == "" {
		return Result{}, errors.New("subagent task is required")
	}
	parentRuntime, err := requireParentRuntime(ctx)
	if err != nil {
		return Result{}, err
	}
	parent := parentRuntime.Capture
	parentScope := parent.Scope()
	childDepth := parentScope.Depth + 1
	if childDepth > topology.config.MaxDepth {
		return Result{}, fmt.Errorf("%w: depth %d exceeds %d", ErrDepthExceeded, childDepth, topology.config.MaxDepth)
	}
	if _, err := agentic.RequireDriver(runner); err != nil {
		return Result{}, err
	}

	var budget harnessruntime.BudgetLease
	if topology.config.Capture.Budget != ModeIsolate {
		budget, err = parent.AcquireBudget(ctx)
		if err != nil {
			return Result{}, err
		}
		defer budget.Close()
	}
	limits := childLimits(topology.config, budget)

	history, err := capturedHistory(ctx, topology.config, parent)
	if err != nil {
		return Result{}, err
	}
	environments, err := capturedEnvironment(parentRuntime, topology.config)
	if err != nil {
		return Result{}, err
	}
	projection := newProjectingFactory(topology.config.Runtime.Events, parent)
	runtimeConfig := topology.config.Runtime
	runtimeConfig.Environments = environments
	runtimeConfig.Events = projection
	runtimeConfig.Scope = harnessruntime.Scope{
		ParentSessionID: parentScope.SessionID,
		Agent:           topology.config.Name,
		Depth:           childDepth,
	}
	childCapabilities := capturedCapabilities(topology, parent, childDepth)
	childHarness, err := harness.New(
		runner,
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(childCapabilities...),
	).Build()
	if err != nil {
		return Result{}, fmt.Errorf("build child harness: %w", err)
	}
	var options []harness.SessionOption
	if len(history) > 0 {
		options = append(options, harness.WithInitialHistory(history...))
	}
	if limits != nil {
		options = append(options, harness.WithBudget(*limits))
	}
	child, err := childHarness.NewSession(ctx, options...)
	if err != nil {
		return Result{}, fmt.Errorf("create child session: %w", err)
	}
	address := Address{ParentSessionID: parentScope.SessionID, ChildSessionID: child.ID()}
	if err := topology.router.add(address, child); err != nil {
		_ = child.Close(context.Background())
		return Result{}, err
	}
	defer topology.router.remove(address, child)
	if err := projection.activate(); err != nil {
		_ = child.Close(context.Background())
		return Result{}, err
	}

	execution, childErr := promptChild(ctx, child, task, runtimeConfig.ToolCancellationGrace)
	snapshot, _ := child.Snapshot(context.Background())
	closeErr := child.Close(context.Background())
	if projectErr := projection.Err(); projectErr != nil {
		return Result{}, projectErr
	}
	if closeErr != nil {
		return Result{}, fmt.Errorf("close child session: %w", closeErr)
	}
	result := summarizeChild(topology.config, child.ID(), execution, childErr, snapshot.Usage)
	if budget != nil {
		if err := budget.Commit(context.WithoutCancel(ctx), harnessruntime.UsageCharge{
			SessionID: child.ID(),
			Agent:     topology.config.Name,
			Usage:     snapshot.Usage,
		}); err != nil {
			return result, err
		}
	}
	return result, nil
}

func capturedHistory(
	ctx context.Context,
	config Config,
	parent harnessruntime.CaptureRuntime,
) ([]agentic.Message, error) {
	switch config.Capture.History {
	case ModeIsolate:
		return nil, nil
	case ModeShare:
		return withoutSystemHistory(parent.History()), nil
	case ModeNarrow:
		history := withoutSystemHistory(parent.History())
		projected, err := config.HistoryProjector(ctx, history)
		if err != nil {
			return nil, fmt.Errorf("project child history: %w", err)
		}
		return withoutSystemHistory(projected), nil
	default:
		return nil, fmt.Errorf("%w: history mode %d", ErrInvalidCapture, config.Capture.History)
	}
}

func withoutSystemHistory(messages []agentic.Message) []agentic.Message {
	result := make([]agentic.Message, 0, len(messages))
	for _, message := range messages {
		if message.Role != agentic.RoleSystem {
			result = append(result, message)
		}
	}
	return cloneMessages(result)
}

func capturedEnvironment(parent harnessruntime.ToolRuntime, config Config) (env.Factory, error) {
	switch config.Capture.Environment {
	case ModeIsolate:
		return config.Runtime.Environments, nil
	case ModeShare:
		return env.FactoryFunc(func(ctx context.Context, _ string) (env.Lease, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			borrowed := borrowedEnvironment{Environment: parent.Environment}
			if narrower, ok := parent.Environment.(env.Narrower); ok {
				return borrowedNarrowEnvironment{
					borrowedEnvironment: borrowed,
					narrower:            narrower,
				}, nil
			}
			return borrowed, nil
		}), nil
	case ModeNarrow:
		narrower, ok := parent.Environment.(env.Narrower)
		if !ok {
			return nil, errors.New("parent environment does not support narrowing")
		}
		request := env.NarrowRequest{
			Root:  config.EnvironmentRoot,
			Shell: config.EnvironmentShell,
		}
		return env.FactoryFunc(func(ctx context.Context, _ string) (env.Lease, error) {
			return narrower.Narrow(ctx, request)
		}), nil
	default:
		return nil, fmt.Errorf("%w: environment mode %d", ErrInvalidCapture, config.Capture.Environment)
	}
}

type borrowedEnvironment struct {
	env.Environment
}

func (borrowedEnvironment) Close(context.Context) error { return nil }

type borrowedNarrowEnvironment struct {
	borrowedEnvironment
	narrower env.Narrower
}

func (b borrowedNarrowEnvironment) Narrow(
	ctx context.Context,
	request env.NarrowRequest,
) (env.Lease, error) {
	return b.narrower.Narrow(ctx, request)
}

func capturedCapabilities(
	topology *Capability,
	parent harnessruntime.CaptureRuntime,
	childDepth int,
) []harness.Capability {
	config := topology.config
	children := append([]harness.Capability(nil), config.Capabilities...)
	before := capabilityIDs(children)
	internal := make([]harness.Capability, 0, 2)
	if config.Capture.Tools != ModeIsolate {
		internal = append(internal, inheritedToolsCapability(topology, parent, childDepth, before))
	}
	if config.Capture.Permissions != ModeIsolate && parent.ToolGate() != nil {
		internal = append(internal, inheritedGateCapability(topology, parent.ToolGate(), before))
	}
	return append(internal, children...)
}

func capabilityIDs(values []harness.Capability) []string {
	result := make([]string, 0, len(values))
	for _, value := range values {
		if value != nil && value.ID() != "" {
			result = append(result, value.ID())
		}
	}
	return result
}

func inheritedToolsCapability(
	topology *Capability,
	parent harnessruntime.CaptureRuntime,
	childDepth int,
	before []string,
) harness.Capability {
	id := "subagent.capture.tools:" + topology.config.Name
	delegation := make(map[string]bool)
	for _, name := range parent.DelegationTools() {
		delegation[name] = true
	}
	allowRecursion := childDepth < topology.config.MaxDepth
	return capability.Func{
		Name:  id,
		Order: capability.Ordering{Before: append([]string(nil), before...)},
		Apply: func(registry *capability.Registry) error {
			for _, toolset := range parent.Toolsets() {
				imported := make([]string, 0)
				filtered := agentic.FilterToolset(toolset, func(name string) bool {
					if delegation[name] && !allowRecursion {
						return false
					}
					if topology.config.Capture.Tools == ModeNarrow && !topology.config.ToolFilter(name) {
						return false
					}
					imported = append(imported, name)
					return true
				})
				if err := registry.AddToolset(filtered); err != nil {
					return err
				}
				for _, name := range imported {
					if delegation[name] {
						if err := registry.MarkDelegationTool(name); err != nil {
							return err
						}
					}
				}
			}
			return nil
		},
	}
}

func inheritedGateCapability(
	topology *Capability,
	gate agentic.ToolGate,
	before []string,
) harness.Capability {
	return capability.Func{
		Name:  "subagent.capture.permissions:" + topology.config.Name,
		Order: capability.Ordering{Before: append([]string(nil), before...)},
		Apply: func(registry *capability.Registry) error {
			return registry.AddToolGateMiddleware(capability.ToolGateMiddlewareFunc(func(
				ctx context.Context,
				calls []agentic.ToolUse,
				_ agentic.ToolBatchDecision,
			) (agentic.ToolBatchDecision, error) {
				return gate.EvaluateBatch(ctx, calls)
			}))
		},
	}
}

func childLimits(config Config, lease harnessruntime.BudgetLease) *agentic.UsageLimits {
	var parent *agentic.UsageLimits
	if lease != nil {
		parent = lease.Limits()
	}
	switch config.Capture.Budget {
	case ModeShare:
		return cloneLimitsPointer(parent)
	case ModeNarrow:
		return intersectLimits(parent, config.Budget)
	case ModeIsolate:
		return cloneLimitsPointer(config.Budget)
	default:
		return nil
	}
}

func intersectLimits(left, right *agentic.UsageLimits) *agentic.UsageLimits {
	if left == nil {
		return cloneLimitsPointer(right)
	}
	if right == nil {
		return cloneLimitsPointer(left)
	}
	return &agentic.UsageLimits{
		MaxRequestTokens:  minLimit(left.MaxRequestTokens, right.MaxRequestTokens),
		MaxResponseTokens: minLimit(left.MaxResponseTokens, right.MaxResponseTokens),
		MaxTotalTokens:    minLimit(left.MaxTotalTokens, right.MaxTotalTokens),
		MaxRequests:       minLimit(left.MaxRequests, right.MaxRequests),
		MaxToolCalls:      minLimit(left.MaxToolCalls, right.MaxToolCalls),
	}
}

func minLimit(left, right *int) *int {
	if left == nil {
		return cloneInt(right)
	}
	if right == nil {
		return cloneInt(left)
	}
	value := min(*left, *right)
	return &value
}

func cloneLimitsPointer(value *agentic.UsageLimits) *agentic.UsageLimits {
	if value == nil {
		return nil
	}
	copy := cloneLimits(*value)
	return &copy
}

type promptResult[O any] struct {
	execution *agentic.Execution[O]
	err       error
}

func promptChild[O any](
	ctx context.Context,
	child *harness.Session[O],
	task string,
	grace time.Duration,
) (*agentic.Execution[O], error) {
	done := make(chan promptResult[O], 1)
	go func() {
		execution, err := child.Prompt(ctx, agentic.NewTextMessage(agentic.RoleUser, task))
		done <- promptResult[O]{execution: execution, err: err}
	}()
	select {
	case result := <-done:
		return result.execution, result.err
	case <-ctx.Done():
		if grace <= 0 {
			grace = time.Second
		}
		interruptCtx, cancel := context.WithTimeout(context.Background(), grace+time.Second)
		defer cancel()
		_ = child.Interrupt(interruptCtx)
		select {
		case result := <-done:
			return result.execution, result.err
		case <-interruptCtx.Done():
			return nil, ctx.Err()
		}
	}
}

func summarizeChild[O any](
	config Config,
	sessionID string,
	execution *agentic.Execution[O],
	runErr error,
	usage agentic.Usage,
) Result {
	status := agentic.ExecutionFailed
	summary := ""
	if execution != nil {
		status = execution.Status
		if execution.Result != nil {
			summary = agentic.FormatToolResult(execution.Result.Output)
		}
	}
	if runErr != nil {
		errorText := runErr.Error()
		if strings.TrimSpace(summary) == "" {
			summary = errorText
		} else {
			summary += "\nChild error: " + errorText
		}
	}
	if strings.TrimSpace(summary) == "" {
		summary = fmt.Sprintf("Child session ended with status %d.", status)
	}
	summary = strings.ToValidUTF8(summary, "\uFFFD")
	fullBytes := len(summary)
	summary, truncated := truncateUTF8(summary, config.SummaryBytes)
	return Result{
		Agent:     config.Name,
		SessionID: sessionID,
		Status:    status,
		Summary:   summary,
		FullBytes: fullBytes,
		Truncated: truncated,
		Usage:     summaryUsage(usage),
	}
}

func truncateUTF8(value string, limit int) (string, bool) {
	if len(value) <= limit {
		return value, false
	}
	end := limit
	for end > 0 && !utf8.RuneStart(value[end]) {
		end--
	}
	return value[:end], true
}

func cloneMessages(messages []agentic.Message) []agentic.Message {
	result := make([]agentic.Message, len(messages))
	copy(result, messages)
	return result
}

func summaryUsage(usage agentic.Usage) agentic.Usage {
	// Per-request detail remains in the durable child transcript. The bounded
	// tool result exposes only fixed-size cumulative counters.
	usage.RequestUsages = nil
	return usage
}
