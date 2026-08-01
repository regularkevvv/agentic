package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	memoryenv "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/permission"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	"github.com/regularkevvv/agentic/harness/store"
	memoryjournal "github.com/regularkevvv/agentic/harness/store/memory"
)

func TestDelegationPersistsSeparateChildAndProjectsBoundedResult(t *testing.T) {
	runtimeConfig, repository := testRuntime(t)
	fullChildOutput := strings.Repeat("é", 64)
	childModel := newModel("child", func(_ context.Context, _ *agentic.ChatRequest, _ int) (*agentic.ChatResponse, error) {
		return textResponse(fullChildOutput, usage(2, 3)), nil
	})
	childRunner := agentic.NewAgent("child system", childModel)
	topology, err := New(childRunner, Config{
		Name:         "researcher",
		Description:  "Research one bounded task.",
		Runtime:      runtimeConfig,
		SummaryBytes: 17,
	})
	if err != nil {
		t.Fatal(err)
	}

	var toolResult string
	parentModel := newModel("parent", func(_ context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		if call == 1 {
			return toolResponse("delegate-1", "researcher", map[string]any{"task": "investigate"}, usage(1, 1)), nil
		}
		for _, result := range request.Messages[len(request.Messages)-1].GetToolResults() {
			toolResult = result.Content
		}
		return textResponse("parent complete", usage(1, 1)), nil
	})
	parentHarness, err := harness.New(
		agentic.NewAgent("parent system", parentModel),
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close(context.Background())
	subscription := parent.Subscribe(event.SubscribeOptions{Buffer: 256, Preview: true})
	defer subscription.Close()

	execution, err := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "delegate this"))
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != agentic.ExecutionCompleted || execution.Result.Output != "parent complete" {
		t.Fatalf("unexpected parent execution %#v", execution)
	}

	var childResult Result
	if err := json.Unmarshal([]byte(toolResult), &childResult); err != nil {
		t.Fatalf("decode child result %q: %v", toolResult, err)
	}
	if childResult.Agent != "researcher" || childResult.SessionID == "" ||
		childResult.Status != agentic.ExecutionCompleted {
		t.Fatalf("unexpected child result %#v", childResult)
	}
	if !childResult.Truncated || childResult.FullBytes != len(fullChildOutput) ||
		len(childResult.Summary) > 17 || !utf8.ValidString(childResult.Summary) {
		t.Fatalf("summary was not UTF-8 bounded: %#v", childResult)
	}

	snapshot, err := parent.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Usage.Requests != 3 {
		t.Fatalf("parent cumulative requests = %d, want parent 2 + child 1", snapshot.Usage.Requests)
	}
	for _, message := range snapshot.Messages {
		if strings.Contains(message.GetTextContent(), fullChildOutput) {
			t.Fatal("full child output leaked into the parent transcript")
		}
	}

	projected := drainEvents(subscription)
	var childEvents int
	for _, record := range projected {
		if record.SessionID != childResult.SessionID {
			continue
		}
		childEvents++
		if record.ParentID != parent.ID() || record.Agent != "researcher" || record.Depth != 1 {
			t.Fatalf("child event was not tagged: %#v", record)
		}
	}
	if childEvents == 0 {
		t.Fatal("no child events were projected onto the parent bus")
	}

	journal, err := repository.Open(context.Background(), childResult.SessionID)
	if err != nil {
		t.Fatalf("open separate child journal: %v", err)
	}
	loaded, err := journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_ = journal.Close(context.Background())
	if len(loaded.Entries) < 4 || childResult.SessionID == parent.ID() {
		t.Fatalf("child did not retain a separate durable transcript: %d entries", len(loaded.Entries))
	}

	childRuntime := runtimeConfig
	childRuntime.Scope.ParentSessionID = parent.ID()
	childRuntime.Scope.Agent = "researcher"
	childRuntime.Scope.Depth = 1
	recoveryHarness, err := harness.New(
		childRunner,
		harness.WithRuntime(childRuntime),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	recoveredChild, err := recoveryHarness.ResumeSession(context.Background(), childResult.SessionID)
	if err != nil {
		t.Fatal(err)
	}
	recoveredSnapshot, err := recoveredChild.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := recoveredChild.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	var recoveredFullOutput bool
	for _, message := range recoveredSnapshot.Messages {
		recoveredFullOutput = recoveredFullOutput || strings.Contains(message.GetTextContent(), fullChildOutput)
	}
	if !recoveredFullOutput {
		t.Fatal("full child output was not recoverable from the child session")
	}
}

func TestAddressedChildRoutingDoesNotConsumeParentSteer(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	childStarted := make(chan struct{})
	releaseChild := make(chan struct{})
	var childMessagesMu sync.Mutex
	var childMessages []string
	childModel := newModel("child", func(ctx context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		childMessagesMu.Lock()
		childMessages = append(childMessages, textMessages(request.Messages)...)
		childMessagesMu.Unlock()
		if call == 1 {
			close(childStarted)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-releaseChild:
			}
			return textResponse("child intermediate", usage(1, 1)), nil
		}
		return textResponse("child steered", usage(1, 1)), nil
	})
	topology, err := New(agentic.NewAgent("", childModel), Config{
		Name:        "delegate",
		Description: "Delegate one task.",
		Runtime:     runtimeConfig,
	})
	if err != nil {
		t.Fatal(err)
	}

	var parentMessagesMu sync.Mutex
	var parentMessages []string
	parentModel := newModel("parent", func(_ context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		parentMessagesMu.Lock()
		parentMessages = append(parentMessages, textMessages(request.Messages)...)
		parentMessagesMu.Unlock()
		switch call {
		case 1:
			return toolResponse("delegate-1", "delegate", map[string]any{"task": "child task"}, usage(1, 1)), nil
		default:
			return textResponse("parent steered", usage(1, 1)), nil
		}
	})
	parentHarness, err := harness.New(
		agentic.NewAgent("", parentModel),
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close(context.Background())
	subscription := parent.Subscribe(event.SubscribeOptions{Buffer: 256})
	defer subscription.Close()

	type promptOutcome struct {
		execution *agentic.Execution[string]
		err       error
	}
	done := make(chan promptOutcome, 1)
	go func() {
		execution, runErr := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "parent task"))
		done <- promptOutcome{execution: execution, err: runErr}
	}()
	select {
	case <-childStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("child model did not start")
	}
	address := waitForChildAddress(t, subscription, parent.ID())
	if _, err := parent.Steer(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "parent-only steer")); err != nil {
		t.Fatal(err)
	}
	if _, err := topology.Steer(context.Background(), address, agentic.NewTextMessage(agentic.RoleUser, "child-only steer")); err != nil {
		t.Fatal(err)
	}
	close(releaseChild)
	select {
	case result := <-done:
		if result.err != nil {
			t.Fatal(result.err)
		}
		if result.execution.Result.Output != "parent steered" {
			t.Fatalf("parent output = %q", result.execution.Result.Output)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("parent execution did not finish")
	}

	childMessagesMu.Lock()
	childJoined := strings.Join(childMessages, "\n")
	childMessagesMu.Unlock()
	if !strings.Contains(childJoined, "child-only steer") || strings.Contains(childJoined, "parent-only steer") {
		t.Fatalf("child routing crossed inboxes:\n%s", childJoined)
	}
	parentMessagesMu.Lock()
	parentJoined := strings.Join(parentMessages, "\n")
	parentMessagesMu.Unlock()
	if !strings.Contains(parentJoined, "parent-only steer") || strings.Contains(parentJoined, "child-only steer") {
		t.Fatalf("parent routing crossed inboxes:\n%s", parentJoined)
	}
	if len(topology.Children(parent.ID())) != 0 {
		t.Fatal("completed child remained in the live router")
	}
}

func TestSharedBudgetStopsParentAndSurvivesRestart(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	childModel := newModel("child", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		return textResponse("child", usage(1, 1)), nil
	})
	topology, err := New(agentic.NewAgent("", childModel), Config{
		Name:        "delegate",
		Description: "Delegate.",
		Runtime:     runtimeConfig,
	})
	if err != nil {
		t.Fatal(err)
	}
	parentModel := newModel("parent", func(_ context.Context, _ *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		if call == 1 {
			return toolResponse("delegate-1", "delegate", map[string]any{"task": "work"}, usage(1, 1)), nil
		}
		return textResponse("must not run", usage(1, 1)), nil
	})
	parentHarness, err := harness.New(
		agentic.NewAgent("", parentModel),
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(context.Background(), harness.WithBudget(agentic.UsageLimits{
		MaxRequests: agentic.IntPtr(2),
	}))
	if err != nil {
		t.Fatal(err)
	}
	_, runErr := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "work"))
	if !errors.Is(runErr, harness.ErrBudgetExceeded) {
		t.Fatalf("expected shared budget exhaustion, got %v", runErr)
	}
	if parentModel.Calls() != 1 || childModel.Calls() != 1 {
		t.Fatalf("requests parent=%d child=%d", parentModel.Calls(), childModel.Calls())
	}
	snapshot, err := parent.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Usage.Requests != 2 {
		t.Fatalf("cumulative usage = %#v", snapshot.Usage)
	}
	id := parent.ID()
	if err := parent.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	recovered, err := parentHarness.ResumeSession(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	defer recovered.Close(context.Background())
	recoveredSnapshot, err := recovered.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if recoveredSnapshot.Usage.Requests != 2 {
		t.Fatalf("recovered cumulative usage = %#v", recoveredSnapshot.Usage)
	}
}

func TestConcurrentDelegationsSerializeSharedBudgetAndCommitExactlyOnce(t *testing.T) {
	runtimeConfig, repository := testRuntime(t)
	var active atomic.Int32
	var maximum atomic.Int32
	childModel := newModel("serialized-child", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		current := active.Add(1)
		for {
			observed := maximum.Load()
			if current <= observed || maximum.CompareAndSwap(observed, current) {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
		active.Add(-1)
		return textResponse("child", usage(1, 1)), nil
	})
	topology, err := New(agentic.NewAgent("", childModel), Config{
		Name:        "delegate",
		Description: "Delegate with a shared budget.",
		Runtime:     runtimeConfig,
	})
	if err != nil {
		t.Fatal(err)
	}

	var childIDs []string
	parentModel := newModel("concurrent-parent", func(_ context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		if call == 1 {
			return &agentic.ChatResponse{
				ID:    "response",
				Model: "test",
				Message: agentic.NewToolUseMessage(
					agentic.ToolUse{ID: "delegate-1", Name: "delegate", Input: map[string]any{"task": "one"}},
					agentic.ToolUse{ID: "delegate-2", Name: "delegate", Input: map[string]any{"task": "two"}},
				),
				Usage:        usage(1, 1),
				FinishReason: agentic.FinishReasonToolCalls,
			}, nil
		}
		for _, message := range request.Messages {
			for _, result := range message.GetToolResults() {
				var child Result
				if err := json.Unmarshal([]byte(result.Content), &child); err != nil {
					t.Fatalf("decode concurrent child result: %v", err)
				}
				childIDs = append(childIDs, child.SessionID)
			}
		}
		return textResponse("done", usage(1, 1)), nil
	})
	parentHarness, err := harness.New(
		agentic.NewAgent("", parentModel),
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(context.Background(), harness.WithBudget(agentic.UsageLimits{
		MaxRequests: agentic.IntPtr(10),
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "two children")); err != nil {
		t.Fatal(err)
	}
	snapshot, err := parent.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	parentID := parent.ID()
	if err := parent.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if childModel.Calls() != 2 || maximum.Load() != 1 {
		t.Fatalf("shared child execution calls=%d max_concurrency=%d", childModel.Calls(), maximum.Load())
	}
	if len(childIDs) != 2 || childIDs[0] == "" || childIDs[0] == childIDs[1] {
		t.Fatalf("child session IDs = %#v", childIDs)
	}
	if snapshot.Usage.Requests != 4 {
		t.Fatalf("parent plus child cumulative usage = %#v", snapshot.Usage)
	}
	journal, err := repository.Open(context.Background(), parentID)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := journal.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := journal.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	var committed int
	for _, entry := range loaded.Entries {
		if entry.Kind == "subagent.usage" {
			committed++
		}
	}
	if committed != 2 {
		t.Fatalf("durable child usage commits = %d, want 2", committed)
	}
}

func TestRecursionIsExplicitAndDepthBounded(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	var sawDelegationToolMu sync.Mutex
	var sawDelegationTool []bool
	childModel := newModel("recursive-child", func(_ context.Context, request *agentic.ChatRequest, _ int) (*agentic.ChatResponse, error) {
		hasDelegate := hasTool(request.Tools, "delegate")
		sawDelegationToolMu.Lock()
		sawDelegationTool = append(sawDelegationTool, hasDelegate)
		sawDelegationToolMu.Unlock()
		if len(request.Messages) > 0 && len(request.Messages[len(request.Messages)-1].GetToolResults()) > 0 {
			return textResponse("branch summary", usage(1, 1)), nil
		}
		if hasDelegate {
			return toolResponse("recursive-1", "delegate", map[string]any{"task": "nested"}, usage(1, 1)), nil
		}
		return textResponse("leaf summary", usage(1, 1)), nil
	})
	topology, err := New(agentic.NewAgent("", childModel), Config{
		Name:        "delegate",
		Description: "Delegate recursively.",
		Runtime:     runtimeConfig,
		MaxDepth:    2,
		Capture: Capture{
			Tools: ModeShare,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	parentModel := newModel("parent", func(_ context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		if call == 1 {
			return toolResponse("delegate-root", "delegate", map[string]any{"task": "root child"}, usage(1, 1)), nil
		}
		if len(request.Messages[len(request.Messages)-1].GetToolResults()) == 0 {
			t.Fatal("parent did not receive child summary")
		}
		return textResponse("done", usage(1, 1)), nil
	})
	parentHarness, err := harness.New(
		agentic.NewAgent("", parentModel),
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close(context.Background())
	subscription := parent.Subscribe(event.SubscribeOptions{Buffer: 512})
	defer subscription.Close()
	if _, err := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "recurse")); err != nil {
		t.Fatal(err)
	}
	sawDelegationToolMu.Lock()
	values := append([]bool(nil), sawDelegationTool...)
	sawDelegationToolMu.Unlock()
	var sawAllowed, sawRemoved bool
	for _, value := range values {
		sawAllowed = sawAllowed || value
		sawRemoved = sawRemoved || !value
	}
	if !sawAllowed || !sawRemoved {
		t.Fatalf("delegation inheritance by depth = %v", values)
	}
	var depthOne, depthTwo bool
	for _, record := range drainEvents(subscription) {
		depthOne = depthOne || record.Depth == 1
		depthTwo = depthTwo || record.Depth == 2
		if record.Depth > 2 {
			t.Fatalf("event escaped maximum depth: %#v", record)
		}
	}
	if !depthOne || !depthTwo {
		t.Fatalf("projected depths one=%v two=%v", depthOne, depthTwo)
	}
}

func TestParentInterruptCancelsChild(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	childStarted := make(chan struct{})
	childCanceled := make(chan struct{})
	childModel := newModel("cancel-child", func(ctx context.Context, _ *agentic.ChatRequest, _ int) (*agentic.ChatResponse, error) {
		close(childStarted)
		<-ctx.Done()
		close(childCanceled)
		return nil, ctx.Err()
	})
	topology, err := New(agentic.NewAgent("", childModel), Config{
		Name:        "delegate",
		Description: "Delegate cancellable work.",
		Runtime:     runtimeConfig,
	})
	if err != nil {
		t.Fatal(err)
	}
	parentModel := newModel("parent", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		return toolResponse("delegate-1", "delegate", map[string]any{"task": "block"}, usage(1, 1)), nil
	})
	parentHarness, err := harness.New(
		agentic.NewAgent("", parentModel),
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close(context.Background())
	done := make(chan error, 1)
	go func() {
		_, runErr := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "cancel"))
		done <- runErr
	}()
	select {
	case <-childStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("child did not start")
	}
	interruptCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := parent.Interrupt(interruptCtx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-childCanceled:
	case <-time.After(5 * time.Second):
		t.Fatal("parent cancellation did not reach child")
	}
	select {
	case runErr := <-done:
		if !errors.Is(runErr, context.Canceled) {
			t.Fatalf("parent prompt error = %v", runErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("parent prompt did not unwind")
	}
	if len(topology.Children(parent.ID())) != 0 {
		t.Fatal("canceled child remained routed")
	}
}

func TestDefaultPermissionCaptureCannotBroadenParentAndIsolationIsExplicit(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	run := func(t *testing.T, permissionMode Mode) int32 {
		t.Helper()
		var effects atomic.Int32
		workTool, workHandler := agentic.MustToolWithContext(
			"work",
			"Perform child work.",
			func(context.Context, struct{}) (string, error) {
				effects.Add(1)
				return "worked", nil
			},
		)
		childTools := capability.Func{
			Name: "child-tools",
			Apply: func(registry *capability.Registry) error {
				return registry.AddToolset(agentic.NewToolset().Add(workTool, workHandler))
			},
		}
		childPolicy, err := permission.New(permission.DecisionAllow)
		if err != nil {
			t.Fatal(err)
		}
		childPermissions, err := permission.NewCapability(
			childPolicy,
			permission.WithID("child-permissions"),
			permission.WithOrdering(capability.Ordering{After: []string{"child-tools"}}),
		)
		if err != nil {
			t.Fatal(err)
		}
		childModel := newModel("permission-child", func(_ context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
			if call == 1 {
				return toolResponse("work-1", "work", map[string]any{}, usage(1, 1)), nil
			}
			if len(request.Messages[len(request.Messages)-1].GetToolResults()) != 1 {
				t.Fatal("child did not receive a work result")
			}
			return textResponse("child complete", usage(1, 1)), nil
		})
		topology, err := New(agentic.NewAgent("", childModel), Config{
			Name:         "delegate",
			Description:  "Delegate permission-scoped work.",
			Runtime:      runtimeConfig,
			Capabilities: []harness.Capability{childTools, childPermissions},
			Capture: Capture{
				Permissions: permissionMode,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		parentPolicy, err := permission.New(
			permission.DecisionDeny,
			permission.Rule{
				Pattern:  "subagent/delegate/agent/**",
				Decision: permission.DecisionAllow,
			},
		)
		if err != nil {
			t.Fatal(err)
		}
		parentPermissions, err := permission.NewCapability(
			parentPolicy,
			permission.WithID("parent-permissions"),
			permission.WithOrdering(capability.Ordering{After: []string{topology.ID()}}),
		)
		if err != nil {
			t.Fatal(err)
		}
		parentModel := newModel("permission-parent", func(_ context.Context, _ *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
			if call == 1 {
				return toolResponse("delegate-1", "delegate", map[string]any{"task": "work"}, usage(1, 1)), nil
			}
			return textResponse("parent complete", usage(1, 1)), nil
		})
		parentHarness, err := harness.New(
			agentic.NewAgent("", parentModel),
			harness.WithRuntime(runtimeConfig),
			harness.WithCapabilities(topology, parentPermissions),
		).Build()
		if err != nil {
			t.Fatal(err)
		}
		parent, err := parentHarness.NewSession(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if _, err := parent.Prompt(
			context.Background(),
			agentic.NewTextMessage(agentic.RoleUser, "delegate"),
		); err != nil {
			t.Fatal(err)
		}
		if err := parent.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
		return effects.Load()
	}

	if effects := run(t, ModeNarrow); effects != 0 {
		t.Fatalf("narrow child broadened the parent denial: effects=%d", effects)
	}
	if effects := run(t, ModeIsolate); effects != 1 {
		t.Fatalf("isolated child did not use its explicit policy: effects=%d", effects)
	}
}

type dependency struct {
	Marker string
}

func TestDependencyBinderAndSharedHistory(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	deps := &dependency{Marker: "alice"}
	var childSawDeps atomic.Bool
	var childTextMu sync.Mutex
	var childText string
	childModel := newModel("deps-child", func(_ context.Context, request *agentic.ChatRequest, _ int) (*agentic.ChatResponse, error) {
		childTextMu.Lock()
		childText = strings.Join(textMessages(request.Messages), "\n")
		childTextMu.Unlock()
		return textResponse("child", usage(1, 1)), nil
	})
	childAgent := agentic.NewAgentWithDepsDynamic[*dependency](func(ctx agentic.RunContext[*dependency]) (string, error) {
		if ctx.Deps == deps {
			childSawDeps.Store(true)
		}
		return "marker=" + ctx.Deps.Marker, nil
	}, childModel)
	topology, err := NewWithDeps(Config{
		Name:        "delegate",
		Description: "Delegate with dependencies.",
		Runtime:     runtimeConfig,
		Capture: Capture{
			History: ModeShare,
		},
	}, func(ctx agentic.RunContext[*dependency], mode Mode) (agentic.Runner[string], error) {
		if mode != ModeShare {
			return nil, errors.New("unexpected dependency capture mode")
		}
		return childAgent.Bind(ctx.Deps), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	parentModel := newModel("deps-parent", func(_ context.Context, _ *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		if call == 1 {
			return toolResponse("delegate-1", "delegate", map[string]any{"task": "child task"}, usage(1, 1)), nil
		}
		return textResponse("done", usage(1, 1)), nil
	})
	parentRunner := agentic.NewAgentWithDeps[*dependency]("", parentModel).Bind(deps)
	parentHarness, err := harness.New(
		parentRunner,
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(
		context.Background(),
		harness.WithInitialHistory(
			agentic.NewTextMessage(agentic.RoleSystem, "parent secret"),
			agentic.NewTextMessage(agentic.RoleUser, "parent history"),
			agentic.NewTextMessage(agentic.RoleAssistant, "parent answer"),
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close(context.Background())
	if _, err := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "current parent task")); err != nil {
		t.Fatal(err)
	}
	childTextMu.Lock()
	text := childText
	childTextMu.Unlock()
	if !childSawDeps.Load() ||
		!strings.Contains(text, "marker=alice") ||
		!strings.Contains(text, "parent history") ||
		!strings.Contains(text, "current parent task") ||
		!strings.Contains(text, "child task") ||
		strings.Contains(text, "parent secret") {
		t.Fatalf("dependency/history capture failed: deps=%v messages=%q", childSawDeps.Load(), text)
	}
}

func TestCaptureValidationAndDepthGuard(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	runner := agentic.NewAgent("", newModel("child", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		return textResponse("ok", usage(1, 1)), nil
	}))
	tests := []Config{
		{Name: "", Description: "x", Runtime: runtimeConfig},
		{Name: "x", Description: "", Runtime: runtimeConfig},
		{Name: "x", Description: "x", Runtime: runtimeConfig, SummaryBytes: -1},
		{Name: "x", Description: "x", Runtime: runtimeConfig, MaxDepth: -1},
		{Name: "x", Description: "x", Runtime: runtimeConfig, Capture: Capture{History: ModeNarrow}},
		{Name: "x", Description: "x", Runtime: runtimeConfig, Capture: Capture{Environment: ModeNarrow}},
		{Name: "x", Description: "x", Runtime: runtimeConfig, Capture: Capture{Tools: ModeNarrow}},
		{Name: "x", Description: "x", Runtime: runtimeConfig, Capture: Capture{Budget: ModeNarrow}},
		{Name: "x", Description: "x", Runtime: runtimeConfig, Budget: &agentic.UsageLimits{MaxRequests: agentic.IntPtr(1)}},
		{Name: "x", Description: "x", Runtime: runtimeConfig, MaxDepth: 2},
	}
	for index, config := range tests {
		if _, err := New(runner, config); err == nil {
			t.Fatalf("invalid config %d was accepted", index)
		}
	}
	if _, err := New[string](nil, Config{}); err == nil {
		t.Fatal("nil runner was accepted")
	}
	if _, err := NewWithDeps[*dependency, string](Config{}, nil); err == nil {
		t.Fatal("nil dependency binder was accepted")
	}
}

type testModel struct {
	name string
	run  func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error)

	mu       sync.Mutex
	requests []*agentic.ChatRequest
}

func newModel(name string, run func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error)) *testModel {
	return &testModel{name: name, run: run}
}

func (m *testModel) Name() string { return m.name }

func (m *testModel) Request(ctx context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.mu.Lock()
	m.requests = append(m.requests, cloneRequest(request))
	call := len(m.requests)
	m.mu.Unlock()
	return m.run(ctx, request, call)
}

func (m *testModel) Calls() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

func cloneRequest(request *agentic.ChatRequest) *agentic.ChatRequest {
	encoded, _ := json.Marshal(request)
	var cloned agentic.ChatRequest
	_ = json.Unmarshal(encoded, &cloned)
	return &cloned
}

func textResponse(text string, value agentic.Usage) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		ID:           "response",
		Model:        "test",
		Message:      agentic.NewTextMessage(agentic.RoleAssistant, text),
		Usage:        value,
		FinishReason: agentic.FinishReasonStop,
	}
}

func toolResponse(id, name string, input map[string]any, value agentic.Usage) *agentic.ChatResponse {
	return &agentic.ChatResponse{
		ID:    "response",
		Model: "test",
		Message: agentic.NewToolUseMessage(agentic.ToolUse{
			ID:    id,
			Name:  name,
			Input: input,
		}),
		Usage:        value,
		FinishReason: agentic.FinishReasonToolCalls,
	}
}

func usage(prompt, completion int) agentic.Usage {
	return agentic.Usage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      prompt + completion,
	}
}

func testRuntime(t *testing.T) (harness.RuntimeConfig, *memoryjournal.Repository) {
	t.Helper()
	repository := memoryjournal.New()
	environment, err := memoryenv.NewFactory(memoryenv.Config{Cwd: "/"})
	if err != nil {
		t.Fatal(err)
	}
	processors, err := spill.NewFactory(artifactmemory.New(), spill.Config{Disabled: true})
	if err != nil {
		t.Fatal(err)
	}
	return harness.RuntimeConfig{
		Sessions:         repository,
		Codec:            jsoncodec.New(),
		Events:           inproc.NewFactory(),
		Environments:     environment,
		ResultProcessors: processors,
		Clock:            system.NewClock(),
		IDs:              system.NewIDs(),
	}, repository
}

func waitForChildAddress(t *testing.T, subscription *event.Subscription, parentID string) Address {
	t.Helper()
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case record := <-subscription.Events:
			if record.Agent != "" && record.SessionID != "" && record.SessionID != parentID {
				return Address{ParentSessionID: parentID, ChildSessionID: record.SessionID}
			}
		case err := <-subscription.Err:
			t.Fatalf("subscription ended before child route: %v", err)
		case <-timeout.C:
			t.Fatal("timed out waiting for child event")
		}
	}
}

func drainEvents(subscription *event.Subscription) []event.Record {
	var result []event.Record
	for {
		select {
		case record, ok := <-subscription.Events:
			if !ok {
				return result
			}
			result = append(result, record)
		default:
			return result
		}
	}
}

func textMessages(messages []agentic.Message) []string {
	var result []string
	for _, message := range messages {
		if text := message.GetTextContent(); text != "" {
			result = append(result, text)
		}
	}
	return result
}

func hasTool(tools []agentic.Tool, name string) bool {
	for _, tool := range tools {
		if tool.Function.Name == name {
			return true
		}
	}
	return false
}

var _ agentic.Model = (*testModel)(nil)
var _ store.Repository = (*memoryjournal.Repository)(nil)
