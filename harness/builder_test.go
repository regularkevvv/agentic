package harness

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/capability"
	environmentcapability "github.com/regularkevvv/agentic/harness/capability/environment"
	"github.com/regularkevvv/agentic/harness/permission"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func TestBuilderValidatesRuntimeDriverAndGraph(t *testing.T) {
	t.Parallel()
	if _, err := New[string](&facadeDriver{}).Build(); err == nil {
		t.Fatal("builder without runtime succeeded")
	}
	if _, err := New[string](runnerOnly{}, WithRuntime(runtimeConfig(t))).Build(); !errors.Is(err, agentic.ErrDriverRequired) {
		t.Fatalf("runner-only error = %v", err)
	}
	cycle := []Capability{
		capability.Func{Name: "one", Order: capability.Ordering{After: []string{"two"}}},
		capability.Func{Name: "two", Order: capability.Ordering{After: []string{"one"}}},
	}
	if _, err := New(
		&facadeDriver{},
		WithRuntime(runtimeConfig(t)),
		WithCapabilities(cycle...),
	).Build(); !errors.Is(err, capability.ErrCapabilityCycle) {
		t.Fatalf("cycle error = %v", err)
	}
	if _, err := New[string](&facadeDriver{}, nil).Build(); err == nil {
		t.Fatal("nil option succeeded")
	}
	if _, err := New(
		&facadeDriver{},
		WithRuntime(runtimeConfig(t)),
		WithCapabilities(nil),
	).Build(); err == nil {
		t.Fatal("nil capability succeeded")
	}
	wantErr := errors.New("option")
	if _, err := New(
		&facadeDriver{},
		func(*builderOptions) error { return wantErr },
		WithRuntime(runtimeConfig(t)),
	).Build(); !errors.Is(err, wantErr) {
		t.Fatalf("option error = %v", err)
	}
	var nilBuilder *Builder[string]
	if _, err := nilBuilder.Build(); err == nil {
		t.Fatal("nil builder succeeded")
	}
}

func TestBuilderFreezesCapabilityOrderAndRunsLifecycleHooks(t *testing.T) {
	t.Parallel()
	var mu sync.Mutex
	var events []string
	hook := func(name string) harnessruntime.LifecycleHook {
		return harnessruntime.LifecycleHookFunc(func(_ context.Context, event harnessruntime.LifecycleEvent) error {
			mu.Lock()
			events = append(events, name+":"+string(event.Phase))
			mu.Unlock()
			return nil
		})
	}
	first := capability.Func{Name: "first", Apply: func(registry *capability.Registry) error {
		return registry.AddLifecycleHook(hook("first"))
	}}
	second := capability.Func{
		Name:  "second",
		Order: capability.Ordering{After: []string{"first"}},
		Apply: func(registry *capability.Registry) error {
			return registry.AddLifecycleHook(hook("second"))
		},
	}
	runtime, err := New(
		&facadeDriver{},
		WithRuntime(runtimeConfig(t)),
		WithCapabilities(second, first),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(runtime.Capabilities(), []string{"first", "second"}) {
		t.Fatalf("capabilities = %v", runtime.Capabilities())
	}
	copy := runtime.Capabilities()
	copy[0] = "changed"
	if reflect.DeepEqual(copy, runtime.Capabilities()) {
		t.Fatal("capability order is mutable")
	}
	session, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"first:session.opened",
		"second:session.opened",
		"first:session.closing",
		"second:session.closing",
		"first:session.closed",
		"second:session.closed",
	}
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("lifecycle events = %v", events)
	}
}

type capabilityModel struct {
	mu       sync.Mutex
	requests []agentic.ChatRequest
}

func (m *capabilityModel) Name() string { return "test:capability-builder" }

func (m *capabilityModel) Request(_ context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	m.mu.Lock()
	copy := *request
	copy.Messages = append([]agentic.Message(nil), request.Messages...)
	copy.Tools = append([]agentic.Tool(nil), request.Tools...)
	m.requests = append(m.requests, copy)
	index := len(m.requests)
	m.mu.Unlock()
	if index == 1 {
		call := agentic.ToolUse{
			ID:   "write-1",
			Name: environmentcapability.ToolWriteFile,
			Input: map[string]any{
				"path":    "/created.txt",
				"content": "created",
			},
		}
		return &agentic.ChatResponse{
			Model:           m.Name(),
			Message:         agentic.NewToolUseMessage(call),
			FinishReason:    agentic.FinishReasonToolCalls,
			RawFinishReason: string(agentic.FinishReasonToolCalls),
		}, nil
	}
	return &agentic.ChatResponse{
		Model:           m.Name(),
		Message:         agentic.NewTextMessage(agentic.RoleAssistant, "done"),
		FinishReason:    agentic.FinishReasonStop,
		RawFinishReason: string(agentic.FinishReasonStop),
	}, nil
}

func TestBuilderPlanReachesDriverAsToolsAndGate(t *testing.T) {
	model := &capabilityModel{}
	agent := agentic.NewAgent("", model)
	environmentCapability, err := environmentcapability.New(environmentcapability.Config{Files: true})
	if err != nil {
		t.Fatal(err)
	}
	permissionCapability, err := permission.NewCapability(
		permission.WorkspaceWrite(),
		permission.WithOrdering(capability.Ordering{After: []string{environmentcapability.ID}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := New(
		agent,
		WithRuntime(runtimeConfig(t)),
		WithCapabilities(environmentCapability, permissionCapability),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	session, err := runtime.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	execution, err := session.Prompt(context.Background(), agentic.NewTextMessage(agentic.RoleUser, "write"))
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != agentic.ExecutionCompleted {
		t.Fatalf("status = %v", execution.Status)
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	if len(model.requests) != 2 || len(model.requests[0].Tools) != 6 {
		t.Fatalf("requests=%d tools=%d", len(model.requests), len(model.requests[0].Tools))
	}
	results := model.requests[1].Messages[len(model.requests[1].Messages)-1].GetToolResults()
	if len(results) != 1 || results[0].IsError {
		t.Fatalf("tool results = %#v", results)
	}
}
