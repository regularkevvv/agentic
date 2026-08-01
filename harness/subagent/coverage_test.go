package subagent

import (
	"context"
	"errors"
	"io/fs"
	"reflect"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	memoryenv "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type captureStub struct {
	history      []agentic.Message
	toolsets     []agentic.Toolset
	gate         agentic.ToolGate
	delegation   []string
	scope        harnessruntime.Scope
	budget       harnessruntime.BudgetLease
	budgetErr    error
	projectErr   error
	projectAfter int

	mu        sync.Mutex
	projected []event.Record
}

func (c *captureStub) History() []agentic.Message {
	return append([]agentic.Message(nil), c.history...)
}
func (c *captureStub) Toolsets() []agentic.Toolset {
	return append([]agentic.Toolset(nil), c.toolsets...)
}
func (c *captureStub) ToolGate() agentic.ToolGate { return c.gate }
func (c *captureStub) DelegationTools() []string {
	return append([]string(nil), c.delegation...)
}
func (c *captureStub) Scope() harnessruntime.Scope { return c.scope }
func (c *captureStub) AcquireBudget(context.Context) (harnessruntime.BudgetLease, error) {
	return c.budget, c.budgetErr
}
func (c *captureStub) ProjectEvent(_ context.Context, record event.Record) error {
	c.mu.Lock()
	c.projected = append(c.projected, record)
	count := len(c.projected)
	c.mu.Unlock()
	if c.projectAfter > 0 && count <= c.projectAfter {
		return nil
	}
	return c.projectErr
}

type budgetStub struct {
	limits    *agentic.UsageLimits
	committed []harnessruntime.UsageCharge
	closed    int
	err       error
}

func (b *budgetStub) Limits() *agentic.UsageLimits { return cloneLimitsPointer(b.limits) }
func (b *budgetStub) Commit(_ context.Context, charge harnessruntime.UsageCharge) error {
	b.committed = append(b.committed, charge)
	return b.err
}
func (b *budgetStub) Close() { b.closed++ }

type childStub struct {
	receipt  harness.QueueReceipt
	snapshot harness.Snapshot
	err      error
}

func (c *childStub) Steer(context.Context, agentic.Message) (harness.QueueReceipt, error) {
	return c.receipt, c.err
}
func (c *childStub) FollowUp(context.Context, agentic.Message) (harness.QueueReceipt, error) {
	return c.receipt, c.err
}
func (c *childStub) NextTurn(context.Context, agentic.Message) (harness.QueueReceipt, error) {
	return c.receipt, c.err
}
func (c *childStub) Interrupt(context.Context) error { return c.err }
func (c *childStub) Snapshot(context.Context) (harness.Snapshot, error) {
	return c.snapshot, c.err
}

type runnerOnly struct{}

func (runnerOnly) Run(context.Context, string, ...agentic.RunOption) (*agentic.Result[string], error) {
	return &agentic.Result[string]{Output: "runner-only"}, nil
}

type environmentStub struct{}

func (environmentStub) Files() env.FileSystem    { return environmentStub{} }
func (environmentStub) Shell() (env.Shell, bool) { return nil, false }
func (environmentStub) CanonicalPath(context.Context, string) (env.CanonicalResource, error) {
	return env.CanonicalResource{Scheme: "stub", ID: "root"}, nil
}
func (environmentStub) ReadFile(context.Context, string) ([]byte, error) {
	return nil, fs.ErrNotExist
}
func (environmentStub) WriteFile(context.Context, string, []byte, fs.FileMode) error {
	return nil
}
func (environmentStub) MkdirAll(context.Context, string, fs.FileMode) error { return nil }
func (environmentStub) ReadDir(context.Context, string) ([]env.DirEntry, error) {
	return nil, nil
}
func (environmentStub) Stat(context.Context, string) (env.FileInfo, error) {
	return env.FileInfo{}, nil
}
func (environmentStub) Remove(context.Context, string) error { return nil }

type leaseStub struct {
	environmentStub
	closeErr error
}

func (l *leaseStub) Close(context.Context) error { return l.closeErr }

type fixedIDs string

func (f fixedIDs) New(string) (string, error) { return string(f), nil }

func TestRouterMethodsAndErrors(t *testing.T) {
	topology := &Capability{router: newRouter()}
	if len(topology.Children("")) != 0 {
		t.Fatal("empty router returned children")
	}
	if (*Capability)(nil).Children("") != nil {
		t.Fatal("nil capability returned children")
	}
	first := &childStub{
		receipt:  harness.QueueReceipt{ID: "queued"},
		snapshot: harness.Snapshot{State: harness.SessionRunning},
	}
	second := &childStub{}
	addressB := Address{ParentSessionID: "parent", ChildSessionID: "b"}
	addressA := Address{ParentSessionID: "parent", ChildSessionID: "a"}
	if err := topology.router.add(addressB, second); err != nil {
		t.Fatal(err)
	}
	if err := topology.router.add(addressA, first); err != nil {
		t.Fatal(err)
	}
	if err := topology.router.add(addressA, first); err == nil {
		t.Fatal("duplicate child route succeeded")
	}
	if got := topology.Children("parent"); !reflect.DeepEqual(got, []Address{addressA, addressB}) {
		t.Fatalf("sorted routes = %#v", got)
	}
	message := agentic.NewTextMessage(agentic.RoleUser, "input")
	if receipt, err := topology.Steer(context.Background(), addressA, message); err != nil || receipt.ID != "queued" {
		t.Fatalf("Steer = %#v, %v", receipt, err)
	}
	if receipt, err := topology.FollowUp(context.Background(), addressA, message); err != nil || receipt.ID != "queued" {
		t.Fatalf("FollowUp = %#v, %v", receipt, err)
	}
	if receipt, err := topology.NextTurn(context.Background(), addressA, message); err != nil || receipt.ID != "queued" {
		t.Fatalf("NextTurn = %#v, %v", receipt, err)
	}
	if err := topology.Interrupt(context.Background(), addressA); err != nil {
		t.Fatal(err)
	}
	if snapshot, err := topology.Snapshot(context.Background(), addressA); err != nil || snapshot.State != harness.SessionRunning {
		t.Fatalf("Snapshot = %#v, %v", snapshot, err)
	}
	for _, operation := range []func() error{
		func() error { _, err := topology.Steer(context.Background(), Address{}, message); return err },
		func() error { _, err := topology.FollowUp(context.Background(), Address{}, message); return err },
		func() error { _, err := topology.NextTurn(context.Background(), Address{}, message); return err },
		func() error { return topology.Interrupt(context.Background(), Address{}) },
		func() error { _, err := topology.Snapshot(context.Background(), Address{}); return err },
	} {
		if err := operation(); !errors.Is(err, ErrChildNotFound) {
			t.Fatalf("missing child route = %v", err)
		}
	}
	if _, err := topology.Steer(
		context.Background(),
		Address{ParentSessionID: "parent", ChildSessionID: "missing"},
		message,
	); !errors.Is(err, ErrChildNotFound) {
		t.Fatalf("complete missing child route = %v", err)
	}
	if _, err := (*Capability)(nil).Steer(context.Background(), addressA, message); !errors.Is(err, ErrChildNotFound) {
		t.Fatalf("nil topology route = %v", err)
	}
	other := Address{ParentSessionID: "another", ChildSessionID: "a"}
	if err := topology.router.add(other, second); err != nil {
		t.Fatal(err)
	}
	if got := topology.Children(""); !reflect.DeepEqual(got, []Address{other, addressA, addressB}) {
		t.Fatalf("cross-parent sorted routes = %#v", got)
	}
	topology.router.remove(addressA, second)
	if len(topology.Children("parent")) != 2 {
		t.Fatal("wrong child instance removed a route")
	}
	topology.router.remove(addressA, first)
	topology.router.remove(other, second)
}

func TestCaptureHelpersCoverEveryMode(t *testing.T) {
	parent := &captureStub{
		history: []agentic.Message{
			agentic.NewTextMessage(agentic.RoleSystem, "parent system"),
			agentic.NewTextMessage(agentic.RoleUser, "history"),
		},
	}
	if history, err := capturedHistory(context.Background(), Config{Capture: Capture{History: ModeIsolate}}, parent); err != nil || history != nil {
		t.Fatalf("isolated history = %#v, %v", history, err)
	}
	if history, err := capturedHistory(context.Background(), Config{Capture: Capture{History: ModeShare}}, parent); err != nil || len(history) != 1 {
		t.Fatalf("shared history = %#v, %v", history, err)
	}
	narrow := Config{
		Capture: Capture{History: ModeNarrow},
		HistoryProjector: func(_ context.Context, messages []agentic.Message) ([]agentic.Message, error) {
			return messages[:1], nil
		},
	}
	if history, err := capturedHistory(context.Background(), narrow, parent); err != nil || len(history) != 1 {
		t.Fatalf("narrow history = %#v, %v", history, err)
	}
	narrow.HistoryProjector = func(_ context.Context, messages []agentic.Message) ([]agentic.Message, error) {
		return append(
			[]agentic.Message{agentic.NewTextMessage(agentic.RoleSystem, "projected system")},
			messages...,
		), nil
	}
	if history, err := capturedHistory(context.Background(), narrow, parent); err != nil ||
		len(history) != 1 || history[0].Role == agentic.RoleSystem {
		t.Fatalf("narrow history leaked a system message: %#v, %v", history, err)
	}
	narrow.HistoryProjector = func(context.Context, []agentic.Message) ([]agentic.Message, error) {
		return nil, errors.New("project")
	}
	if _, err := capturedHistory(context.Background(), narrow, parent); err == nil {
		t.Fatal("history projector error was hidden")
	}
	if _, err := capturedHistory(context.Background(), Config{Capture: Capture{History: Mode(99)}}, parent); err == nil {
		t.Fatal("invalid history mode succeeded")
	}

	memory, err := memoryenv.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer memory.Close(context.Background())
	if err := memory.MkdirAll(context.Background(), "/child", 0o755); err != nil {
		t.Fatal(err)
	}
	parentRuntime := harnessruntime.ToolRuntime{Environment: memory}
	isolateFactory := env.FactoryFunc(func(context.Context, string) (env.Lease, error) {
		return memoryenv.New("/", nil)
	})
	isolated, err := capturedEnvironment(parentRuntime, Config{
		Capture: Capture{Environment: ModeIsolate},
		Runtime: harness.RuntimeConfig{Environments: isolateFactory},
	})
	if err != nil || reflect.ValueOf(isolated).Pointer() != reflect.ValueOf(isolateFactory).Pointer() {
		t.Fatalf("isolated environment = %T, %v", isolated, err)
	}
	shared, err := capturedEnvironment(parentRuntime, Config{Capture: Capture{Environment: ModeShare}})
	if err != nil {
		t.Fatal(err)
	}
	sharedLease, err := shared.Open(context.Background(), "child")
	if err != nil || sharedLease.Files() != memory.Files() {
		t.Fatalf("shared environment = %T, %v", sharedLease, err)
	}
	sharedNarrower, ok := sharedLease.(env.Narrower)
	if !ok {
		t.Fatal("shared environment hid the parent's narrowing capability")
	}
	sharedNested, err := sharedNarrower.Narrow(context.Background(), env.NarrowRequest{Root: "/child"})
	if err != nil {
		t.Fatal(err)
	}
	if err := sharedNested.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := sharedLease.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := shared.Open(canceled, "child"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled shared environment open = %v", err)
	}
	narrowed, err := capturedEnvironment(parentRuntime, Config{
		Capture:         Capture{Environment: ModeNarrow},
		EnvironmentRoot: "/child",
	})
	if err != nil {
		t.Fatal(err)
	}
	narrowLease, err := narrowed.Open(context.Background(), "child")
	if err != nil {
		t.Fatal(err)
	}
	_ = narrowLease.Close(context.Background())
	if _, err := capturedEnvironment(harnessruntime.ToolRuntime{Environment: environmentStub{}}, Config{
		Capture: Capture{Environment: ModeNarrow},
	}); err == nil {
		t.Fatal("unsupported environment narrowing succeeded")
	}
	if _, err := capturedEnvironment(parentRuntime, Config{Capture: Capture{Environment: Mode(99)}}); err == nil {
		t.Fatal("invalid environment mode succeeded")
	}
}

func TestLimitAndCapabilityHelpers(t *testing.T) {
	left := &agentic.UsageLimits{
		MaxRequests:      agentic.IntPtr(5),
		MaxTotalTokens:   agentic.IntPtr(100),
		MaxRequestTokens: agentic.IntPtr(80),
	}
	right := &agentic.UsageLimits{
		MaxRequests:       agentic.IntPtr(3),
		MaxTotalTokens:    agentic.IntPtr(200),
		MaxResponseTokens: agentic.IntPtr(40),
	}
	intersection := intersectLimits(left, right)
	if *intersection.MaxRequests != 3 || *intersection.MaxTotalTokens != 100 ||
		*intersection.MaxRequestTokens != 80 || *intersection.MaxResponseTokens != 40 {
		t.Fatalf("limit intersection = %#v", intersection)
	}
	if intersectLimits(nil, right) == right || intersectLimits(left, nil) == left {
		t.Fatal("limit intersection did not clone")
	}
	if minLimit(nil, nil) != nil || *minLimit(agentic.IntPtr(2), nil) != 2 ||
		*minLimit(nil, agentic.IntPtr(4)) != 4 || *minLimit(agentic.IntPtr(5), agentic.IntPtr(7)) != 5 {
		t.Fatal("minimum limit branches failed")
	}
	lease := &budgetStub{limits: left}
	for _, test := range []struct {
		mode Mode
		cfg  *agentic.UsageLimits
		want *agentic.UsageLimits
	}{
		{ModeShare, nil, left},
		{ModeNarrow, right, intersection},
		{ModeIsolate, right, right},
		{Mode(99), nil, nil},
	} {
		got := childLimits(Config{Capture: Capture{Budget: test.mode}, Budget: test.cfg}, lease)
		if !reflect.DeepEqual(got, test.want) {
			t.Fatalf("child limits mode %d = %#v, want %#v", test.mode, got, test.want)
		}
	}
	if childLimits(Config{Capture: Capture{Budget: ModeShare}}, nil) != nil {
		t.Fatal("unbudgeted shared child received limits")
	}

	values := []harness.Capability{nil, capability.Func{}, capability.Func{Name: "one"}}
	if got := capabilityIDs(values); !reflect.DeepEqual(got, []string{"one"}) {
		t.Fatalf("capability IDs = %#v", got)
	}
	if len(cloneMessages(nil)) != 0 || len(cloneMessages(parentMessages())) != 1 {
		t.Fatal("message clone branches failed")
	}
}

func TestInheritedToolCaptureNarrowsAndRejectsDuplicateImports(t *testing.T) {
	normalTool, normalHandler := agentic.MustToolWithContext(
		"normal",
		"normal",
		func(context.Context, struct{}) (string, error) { return "normal", nil },
	)
	delegateTool, delegateHandler := agentic.MustToolWithContext(
		"delegate",
		"delegate",
		func(context.Context, struct{}) (string, error) { return "delegate", nil },
	)
	parent := &captureStub{
		toolsets: []agentic.Toolset{
			agentic.NewToolset().
				Add(normalTool, normalHandler).
				Add(delegateTool, delegateHandler),
		},
		delegation: []string{"delegate"},
	}
	topology := &Capability{config: Config{
		Name:     "worker",
		MaxDepth: 2,
		Capture: Capture{
			Tools: ModeNarrow,
		},
		ToolFilter: func(name string) bool { return name == "normal" },
	}}
	plan, err := capability.Compile(inheritedToolsCapability(topology, parent, 1, nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Tools()) != 1 || plan.Tools()[0].Function.Name != "normal" ||
		len(plan.DelegationTools()) != 0 {
		t.Fatalf("narrow inherited tools = %#v / %#v", plan.Tools(), plan.DelegationTools())
	}

	parent.toolsets = append(parent.toolsets, agentic.NewToolset().Add(normalTool, normalHandler))
	if _, err := capability.Compile(inheritedToolsCapability(topology, parent, 1, nil)); !errors.Is(err, capability.ErrDuplicateTool) {
		t.Fatalf("duplicate inherited toolsets = %v", err)
	}
}

func TestConfigValidationBranches(t *testing.T) {
	base := Config{Name: "agent", Description: "desc", MaxDepth: 1, SummaryBytes: 1}
	fields := []struct {
		name   string
		mutate func(*Config)
	}{
		{"history mode", func(c *Config) { c.Capture.History = Mode(99) }},
		{"dependencies mode", func(c *Config) { c.Capture.Dependencies = Mode(99) }},
		{"environment mode", func(c *Config) { c.Capture.Environment = Mode(99) }},
		{"tools mode", func(c *Config) { c.Capture.Tools = Mode(99) }},
		{"permissions mode", func(c *Config) { c.Capture.Permissions = Mode(99) }},
		{"budget mode", func(c *Config) { c.Capture.Budget = Mode(99) }},
	}
	for _, test := range fields {
		t.Run(test.name, func(t *testing.T) {
			config := base
			config.Capture = resolveCapture(config.Capture)
			test.mutate(&config)
			if err := validateCapture(config); !errors.Is(err, ErrInvalidCapture) {
				t.Fatalf("validation = %v", err)
			}
		})
	}
	for name, mutate := range map[string]func(*agentic.UsageLimits){
		"request":  func(l *agentic.UsageLimits) { l.MaxRequestTokens = agentic.IntPtr(0) },
		"response": func(l *agentic.UsageLimits) { l.MaxResponseTokens = agentic.IntPtr(0) },
		"total":    func(l *agentic.UsageLimits) { l.MaxTotalTokens = agentic.IntPtr(0) },
		"requests": func(l *agentic.UsageLimits) { l.MaxRequests = agentic.IntPtr(0) },
		"tools":    func(l *agentic.UsageLimits) { l.MaxToolCalls = agentic.IntPtr(0) },
	} {
		t.Run(name, func(t *testing.T) {
			var limits agentic.UsageLimits
			mutate(&limits)
			if err := validateLimits(limits); err == nil {
				t.Fatal("non-positive budget was accepted")
			}
		})
	}
	if err := validateLimits(agentic.UsageLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveConfig(Config{
		Name:         "agent",
		Description:  "desc",
		MaxDepth:     1,
		SummaryBytes: 1,
		Capture:      Capture{Budget: ModeNarrow},
		Budget:       &agentic.UsageLimits{MaxRequests: agentic.IntPtr(0)},
	}); err == nil {
		t.Fatal("invalid configured budget was accepted")
	}
	config := Config{
		Name:         " agent ",
		Description:  " desc ",
		MaxDepth:     1,
		SummaryBytes: 1,
		Capture:      Capture{Budget: ModeNarrow},
		Ordering: capability.Ordering{
			Before: []string{"before"},
			After:  []string{"after"},
		},
		Budget: &agentic.UsageLimits{MaxRequests: agentic.IntPtr(2)},
	}
	resolved, err := resolveConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	config.Ordering.Before[0] = "mutated"
	*config.Budget.MaxRequests = 9
	if resolved.Name != "agent" || resolved.Description != "desc" ||
		resolved.Ordering.Before[0] != "before" || *resolved.Budget.MaxRequests != 2 {
		t.Fatalf("resolved config was not frozen: %#v", resolved)
	}
}

func TestCapabilityAndRuntimeErrorBranches(t *testing.T) {
	if (*Capability)(nil).ID() != "" || !reflect.DeepEqual((*Capability)(nil).Ordering(), capability.Ordering{}) {
		t.Fatal("nil capability identity branches failed")
	}
	if err := (*Capability)(nil).Register(nil); err == nil {
		t.Fatal("nil capability registered")
	}
	if err := (&Capability{}).Register(nil); err == nil {
		t.Fatal("uninitialized capability registered")
	}
	if _, err := requireParentRuntime(context.Background()); err == nil {
		t.Fatal("missing ToolRuntime succeeded")
	}
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environmentStub{},
		SessionID:   "parent",
		Scope:       harnessruntime.Scope{SessionID: "other"},
		Capture:     &captureStub{},
	})
	if _, err := requireParentRuntime(ctx); err == nil {
		t.Fatal("inconsistent ToolRuntime scope succeeded")
	}

	runtimeConfig, _ := testRuntime(t)
	model := newModel("child", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		return textResponse("ok", usage(1, 1)), nil
	})
	topology, err := New(agentic.NewAgent("", model), Config{
		Name:        "delegate",
		Description: "delegate",
		Runtime:     runtimeConfig,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runChild(context.Background(), topology, "", agentic.NewAgent("", model)); err == nil {
		t.Fatal("empty child task succeeded")
	}
	if _, err := runChild(context.Background(), topology, "task", agentic.NewAgent("", model)); err == nil ||
		!strings.Contains(err.Error(), "ToolRuntime") {
		t.Fatalf("missing parent runtime = %v", err)
	}
	parentEnv, _ := memoryenv.New("/", nil)
	defer parentEnv.Close(context.Background())
	parent := &captureStub{scope: harnessruntime.Scope{SessionID: "parent", Depth: 1}}
	parentCtx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: parentEnv,
		SessionID:   "parent",
		Scope:       parent.scope,
		Capture:     parent,
	})
	if _, err := runChild(parentCtx, topology, "task", agentic.NewAgent("", model)); !errors.Is(err, ErrDepthExceeded) {
		t.Fatalf("depth guard = %v", err)
	}
	parent.scope.Depth = 0
	if _, err := runChild(parentCtx, topology, "task", runnerOnly{}); !errors.Is(err, agentic.ErrDriverRequired) {
		t.Fatalf("runner-only child = %v", err)
	}
	parent.budgetErr = errors.New("budget")
	if _, err := runChild(parentCtx, topology, "task", agentic.NewAgent("", model)); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("budget acquisition error = %v", err)
	}
	parent.budgetErr = nil
	topology.config.Capture.Budget = ModeIsolate
	topology.config.Capture.History = ModeNarrow
	topology.config.HistoryProjector = func(context.Context, []agentic.Message) ([]agentic.Message, error) {
		return nil, errors.New("history")
	}
	if _, err := runChild(parentCtx, topology, "task", agentic.NewAgent("", model)); err == nil || !strings.Contains(err.Error(), "history") {
		t.Fatalf("history error = %v", err)
	}
	topology.config.Capture.History = ModeIsolate
	topology.config.Capture.Environment = ModeNarrow
	parentCtx = harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environmentStub{},
		SessionID:   "parent",
		Scope:       parent.scope,
		Capture:     parent,
	})
	if _, err := runChild(parentCtx, topology, "task", agentic.NewAgent("", model)); err == nil ||
		!strings.Contains(err.Error(), "narrow") {
		t.Fatalf("environment narrowing error = %v", err)
	}
}

func TestConstructorsRegistrationAndDependencyBinderBranches(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	model := newModel("child", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		return textResponse("ok", usage(1, 1)), nil
	})
	config := Config{Name: "delegate", Description: "delegate", Runtime: runtimeConfig}
	if _, err := New[string](runnerOnly{}, config); !errors.Is(err, agentic.ErrDriverRequired) {
		t.Fatalf("runner-only constructor = %v", err)
	}
	if _, err := NewWithDeps[*dependency, string](
		Config{},
		func(agentic.RunContext[*dependency], Mode) (agentic.Runner[string], error) {
			return agentic.NewAgent("", model), nil
		},
	); err == nil {
		t.Fatal("invalid dependency-aware config succeeded")
	}

	bindErr := errors.New("bind")
	boundMode := ModeDefault
	binding, err := NewWithDeps[*dependency, string](
		config,
		func(_ agentic.RunContext[*dependency], mode Mode) (agentic.Runner[string], error) {
			boundMode = mode
			return nil, bindErr
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	runBinding := func(topology *Capability) string {
		t.Helper()
		var toolError string
		parentModel := newModel("binding-parent", func(_ context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
			if call == 1 {
				return toolResponse("delegate", "delegate", map[string]any{"task": "work"}, usage(1, 1)), nil
			}
			results := request.Messages[len(request.Messages)-1].GetToolResults()
			if len(results) == 1 {
				toolError = results[0].Content
			}
			return textResponse("done", usage(1, 1)), nil
		})
		parentRunner := agentic.NewAgentWithDeps[*dependency]("", parentModel).Bind(&dependency{})
		parentHarness, buildErr := harness.New(
			parentRunner,
			harness.WithRuntime(runtimeConfig),
			harness.WithCapabilities(topology),
		).Build()
		if buildErr != nil {
			t.Fatal(buildErr)
		}
		parent, openErr := parentHarness.NewSession(context.Background())
		if openErr != nil {
			t.Fatal(openErr)
		}
		defer parent.Close(context.Background())
		if _, promptErr := parent.Prompt(
			context.Background(),
			agentic.NewTextMessage(agentic.RoleUser, "delegate"),
		); promptErr != nil {
			t.Fatal(promptErr)
		}
		return toolError
	}
	if toolError := runBinding(binding); !strings.Contains(toolError, bindErr.Error()) || boundMode != ModeShare {
		t.Fatalf("binder error/mode = %q / %v", toolError, boundMode)
	}

	nilBinding, err := NewWithDeps[*dependency, string](
		config,
		func(agentic.RunContext[*dependency], Mode) (agentic.Runner[string], error) {
			return nil, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if toolError := runBinding(nilBinding); !strings.Contains(toolError, "returned nil") {
		t.Fatalf("nil dependency binding = %q", toolError)
	}

	topology, err := New(agentic.NewAgent("", model), config)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(topology)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.DelegationTools(), []string{"delegate"}) || len(plan.Tools()) != 1 {
		t.Fatalf("compiled subagent plan = %#v / %#v", plan.DelegationTools(), plan.Tools())
	}

	tool, handler := topology.toolset.ToolsAndHandlers()
	duplicate, err := New(agentic.NewAgent("", model), Config{
		Name:        "delegate",
		Description: "delegate",
		Runtime:     runtimeConfig,
		Ordering:    capability.Ordering{After: []string{"existing"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := capability.Compile(
		capability.Func{Name: "existing", Apply: func(registry *capability.Registry) error {
			return registry.AddToolset(agentic.NewToolset().Add(tool[0], handler[0]))
		}},
		duplicate,
	); !errors.Is(err, capability.ErrDuplicateTool) {
		t.Fatalf("subagent duplicate tool registration = %v", err)
	}

	effectDuplicate, err := New(agentic.NewAgent("", model), Config{
		Name:        "delegate",
		Description: "delegate",
		Runtime:     runtimeConfig,
		Ordering:    capability.Ordering{After: []string{"effect"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := capability.Compile(
		capability.Func{Name: "effect", Apply: func(registry *capability.Registry) error {
			return registry.AddEffectResolver("delegate", capability.EffectResolverFunc(func(
				context.Context,
				agentic.ToolUse,
				env.Environment,
			) (capability.Effect, error) {
				return capability.Effect{}, nil
			}))
		}},
		effectDuplicate,
	); !errors.Is(err, capability.ErrDuplicateEffect) {
		t.Fatalf("subagent duplicate effect registration = %v", err)
	}
}

func TestSubagentEffectResolverRejectsEmptyTaskAndAllowsCanonicalTask(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	childModel := newModel("child", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		return textResponse("child complete", usage(1, 1)), nil
	})
	topology, err := New(agentic.NewAgent("", childModel), Config{
		Name:        "delegate",
		Description: "delegate",
		Runtime:     runtimeConfig,
	})
	if err != nil {
		t.Fatal(err)
	}
	policy, err := permission.New(permission.DecisionAllow)
	if err != nil {
		t.Fatal(err)
	}
	permissions, err := permission.NewCapability(
		policy,
		permission.WithOrdering(capability.Ordering{After: []string{topology.ID()}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	parentModel := newModel("parent", func(_ context.Context, request *agentic.ChatRequest, call int) (*agentic.ChatResponse, error) {
		switch call {
		case 1:
			return toolResponse("empty", "delegate", map[string]any{"task": ""}, usage(1, 1)), nil
		case 2:
			results := request.Messages[len(request.Messages)-1].GetToolResults()
			if len(results) != 1 || !results[0].IsError {
				t.Fatalf("empty task was not denied: %#v", results)
			}
			return toolResponse("valid", "delegate", map[string]any{"task": "work"}, usage(1, 1)), nil
		default:
			return textResponse("done", usage(1, 1)), nil
		}
	})
	parentHarness, err := harness.New(
		agentic.NewAgent("", parentModel),
		harness.WithRuntime(runtimeConfig),
		harness.WithCapabilities(topology, permissions),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	parent, err := parentHarness.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close(context.Background())
	if _, err := parent.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "delegate")); err != nil {
		t.Fatal(err)
	}
	if childModel.Calls() != 1 {
		t.Fatalf("child calls = %d, want only the canonical valid task", childModel.Calls())
	}
}

func TestRunChildBuildOpenProjectionCloseAndBudgetCommitErrors(t *testing.T) {
	runtimeConfig, _ := testRuntime(t)
	model := newModel("child", func(context.Context, *agentic.ChatRequest, int) (*agentic.ChatResponse, error) {
		return textResponse("child", usage(1, 1)), nil
	})
	parentEnvironment, err := memoryenv.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer parentEnvironment.Close(context.Background())
	parent := &captureStub{scope: harnessruntime.Scope{SessionID: "parent"}}
	parentContext := func() context.Context {
		return harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
			Environment: parentEnvironment,
			SessionID:   "parent",
			Scope:       parent.scope,
			Capture:     parent,
		})
	}

	buildConfig := Config{
		Name:        "build",
		Description: "build",
		Runtime:     runtimeConfig,
		Capture: Capture{
			Budget: ModeIsolate,
		},
	}
	buildTopology, err := New(agentic.NewAgent("", model), buildConfig)
	if err != nil {
		t.Fatal(err)
	}
	buildTopology.config.Runtime.Sessions = nil
	if _, err := runChild(parentContext(), buildTopology, "task", agentic.NewAgent("", model)); err == nil ||
		!strings.Contains(err.Error(), "repository") {
		t.Fatalf("child build error = %v", err)
	}

	routeConfig := buildConfig
	routeConfig.Name = "route"
	routeConfig.Runtime.IDs = fixedIDs("fixed-child")
	routeTopology, err := New(agentic.NewAgent("", model), routeConfig)
	if err != nil {
		t.Fatal(err)
	}
	routeAddress := Address{ParentSessionID: "parent", ChildSessionID: "fixed-child"}
	existingChild := &childStub{}
	if err := routeTopology.router.add(routeAddress, existingChild); err != nil {
		t.Fatal(err)
	}
	if _, err := runChild(parentContext(), routeTopology, "task", agentic.NewAgent("", model)); err == nil ||
		!strings.Contains(err.Error(), "route already exists") {
		t.Fatalf("duplicate live child route = %v", err)
	}
	routeTopology.router.remove(routeAddress, existingChild)

	openConfig := buildConfig
	openConfig.Name = "open"
	openConfig.Runtime.Events = event.FactoryFunc(func(context.Context, []event.Record) (event.Hub, error) {
		return nil, errors.New("open child events")
	})
	openTopology, err := New(agentic.NewAgent("", model), openConfig)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runChild(parentContext(), openTopology, "task", agentic.NewAgent("", model)); err == nil ||
		!strings.Contains(err.Error(), "create child session") {
		t.Fatalf("child open error = %v", err)
	}

	projectConfig := buildConfig
	projectConfig.Name = "project"
	projectTopology, err := New(agentic.NewAgent("", model), projectConfig)
	if err != nil {
		t.Fatal(err)
	}
	parent.projectErr = errors.New("project child event")
	if _, err := runChild(parentContext(), projectTopology, "task", agentic.NewAgent("", model)); !errors.Is(err, parent.projectErr) {
		t.Fatalf("child projection error = %v", err)
	}
	parent.mu.Lock()
	parent.projected = nil
	parent.mu.Unlock()
	parent.projectAfter = 1
	parent.projectErr = errors.New("project child event after activation")
	if _, err := runChild(parentContext(), projectTopology, "task", agentic.NewAgent("", model)); !errors.Is(err, parent.projectErr) {
		t.Fatalf("late child projection error = %v", err)
	}
	parent.projectAfter = 0
	parent.projectErr = nil

	closeConfig := buildConfig
	closeConfig.Name = "close"
	closeConfig.Capture.Environment = ModeIsolate
	closeConfig.Runtime.Environments = env.FactoryFunc(func(context.Context, string) (env.Lease, error) {
		return &leaseStub{closeErr: errors.New("close child environment")}, nil
	})
	closeTopology, err := New(agentic.NewAgent("", model), closeConfig)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runChild(parentContext(), closeTopology, "task", agentic.NewAgent("", model)); err == nil ||
		!strings.Contains(err.Error(), "close child session") {
		t.Fatalf("child close error = %v", err)
	}

	budget := &budgetStub{err: errors.New("commit child budget")}
	parent.budget = budget
	budgetConfig := Config{Name: "budget", Description: "budget", Runtime: runtimeConfig}
	budgetTopology, err := New(agentic.NewAgent("", model), budgetConfig)
	if err != nil {
		t.Fatal(err)
	}
	result, err := runChild(parentContext(), budgetTopology, "task", agentic.NewAgent("", model))
	if !errors.Is(err, budget.err) || result.SessionID == "" || budget.closed != 1 || len(budget.committed) != 1 {
		t.Fatalf("child budget commit = %#v, %v, budget=%#v", result, err, budget)
	}
}

func TestProjectingFactoryPreviewAndStickyError(t *testing.T) {
	parent := &captureStub{projectErr: errors.New("project")}
	factory := newProjectingFactory(nil, parent)
	if _, err := factory.Open(context.Background(), nil); err == nil {
		t.Fatal("nil event factory opened")
	}
	baseErr := errors.New("open")
	factory = newProjectingFactory(event.FactoryFunc(func(context.Context, []event.Record) (event.Hub, error) {
		return nil, baseErr
	}), parent)
	if _, err := factory.Open(context.Background(), nil); !errors.Is(err, baseErr) {
		t.Fatalf("base open error = %v", err)
	}
	factory = newProjectingFactory(event.FactoryFunc(func(context.Context, []event.Record) (event.Hub, error) {
		return nil, nil
	}), parent)
	if _, err := factory.Open(context.Background(), nil); err == nil ||
		!strings.Contains(err.Error(), "returned nil") {
		t.Fatalf("nil child event hub = %v", err)
	}
	factory = newProjectingFactory(inproc.NewFactory(), parent)
	hub, err := factory.Open(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	hub.PublishPreview(event.Record{Nature: agentic.EventPreview, SessionID: "child"})
	parent.mu.Lock()
	buffered := len(parent.projected)
	parent.mu.Unlock()
	if buffered != 0 {
		t.Fatal("child event projected before route activation")
	}
	if err := factory.activate(); !errors.Is(err, parent.projectErr) {
		t.Fatalf("activation error = %v", err)
	}
	if err := factory.activate(); !errors.Is(err, parent.projectErr) {
		t.Fatalf("sticky activation error = %v", err)
	}
	if !errors.Is(factory.Err(), parent.projectErr) {
		t.Fatalf("projection error = %v", factory.Err())
	}
	parent.mu.Lock()
	count := len(parent.projected)
	parent.mu.Unlock()
	hub.PublishDurable(event.Record{Cursor: 1, Nature: agentic.EventLifecycle, SessionID: "child"})
	parent.mu.Lock()
	after := len(parent.projected)
	parent.mu.Unlock()
	if count != 1 || after != 1 {
		t.Fatalf("sticky projection called parent %d then %d times", count, after)
	}
	hub.Close()
}

func TestSummaryFallbackAndRunErrorBranches(t *testing.T) {
	config := Config{Name: "worker", SummaryBytes: 1024}
	fallback := summarizeChild[string](
		config,
		"child",
		&agentic.Execution[string]{Status: agentic.ExecutionSuspended},
		nil,
		agentic.Usage{Requests: 1, RequestUsages: []agentic.RequestUsage{{}}},
	)
	if !strings.Contains(fallback.Summary, "status") ||
		fallback.Status != agentic.ExecutionSuspended ||
		fallback.Usage.Requests != 1 ||
		fallback.Usage.RequestUsages != nil {
		t.Fatalf("fallback child summary = %#v", fallback)
	}
	runErr := errors.New("child failed")
	failed := summarizeChild[string](config, "child", nil, runErr, agentic.Usage{})
	if failed.Summary != runErr.Error() {
		t.Fatalf("failed child summary = %#v", failed)
	}
	outputAndError := summarizeChild(
		config,
		"child",
		&agentic.Execution[string]{
			Status: agentic.ExecutionFailed,
			Result: &agentic.Result[string]{Output: "partial"},
		},
		runErr,
		agentic.Usage{},
	)
	if !strings.Contains(outputAndError.Summary, "partial") ||
		!strings.Contains(outputAndError.Summary, runErr.Error()) {
		t.Fatalf("child output/error summary = %#v", outputAndError)
	}
	invalidUTF8 := summarizeChild(
		Config{Name: "worker", SummaryBytes: 7},
		"child",
		&agentic.Execution[string]{
			Status: agentic.ExecutionCompleted,
			Result: &agentic.Result[string]{Output: "ok\xffafter"},
		},
		nil,
		agentic.Usage{},
	)
	if !utf8.ValidString(invalidUTF8.Summary) || !invalidUTF8.Truncated ||
		invalidUTF8.FullBytes != len(strings.ToValidUTF8("ok\xffafter", "\uFFFD")) {
		t.Fatalf("invalid UTF-8 child summary = %#v", invalidUTF8)
	}
}

func parentMessages() []agentic.Message {
	return []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "message")}
}

var _ harnessruntime.CaptureRuntime = (*captureStub)(nil)
var _ harnessruntime.BudgetLease = (*budgetStub)(nil)
var _ childControl = (*childStub)(nil)
var _ agentic.Runner[string] = runnerOnly{}
