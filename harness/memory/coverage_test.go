package memory

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type storeOnly struct{ store *fakeStore }

type listFailureStore struct {
	*fakeStore
	listErr error
}

func (s *listFailureStore) List(context.Context, Scope, ListOptions) ([]string, error) {
	return nil, s.listErr
}

func (s storeOnly) Read(ctx context.Context, scope Scope, path string, options ReadOptions) (File, error) {
	return s.store.Read(ctx, scope, path, options)
}

func (s storeOnly) List(ctx context.Context, scope Scope, options ListOptions) ([]string, error) {
	return s.store.List(ctx, scope, options)
}

func (s storeOnly) Mutate(ctx context.Context, scope Scope, mutation Mutation) (MutationResult, error) {
	return s.store.Mutate(ctx, scope, mutation)
}

func capabilityHandlers(t *testing.T, config Config) map[string]agentic.ToolHandler {
	t.Helper()
	value, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	result := make(map[string]agentic.ToolHandler)
	for _, toolset := range plan.Toolsets() {
		tools, handlers := toolset.ToolsAndHandlers()
		for index, tool := range tools {
			result[tool.Function.Name] = handlers[index]
		}
	}
	return result
}

func executeMemoryHandler(
	t *testing.T,
	handler agentic.ToolHandler,
	ctx context.Context,
	callID, name string,
	input map[string]any,
) error {
	t.Helper()
	ctx = agentic.WithToolCallContext(ctx, agentic.ToolCallContext{ID: callID, Name: name, Attempt: 1})
	_, err := handler.Execute(ctx, input, nil)
	return err
}

func TestCapabilityToolFailureFrontiers(t *testing.T) {
	backend := &fakeStore{file: File{Path: "main", Content: []byte("content"), Version: "v1"}}
	resolver := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "tenant", nil })
	handlers := capabilityHandlers(t, Config{
		Store: backend,
		Scope: resolver,
		Limits: CapabilityLimits{
			MaxReadBytes: 8, MaxWriteBytes: 4, MaxListEntries: 2,
			MaxSearchBytes: 8, MaxSearchMatches: 2,
		},
	})
	bare := context.Background()
	runtimeContext := harnessruntime.WithContext(bare, harnessruntime.ToolRuntime{
		SessionID: "session", Scope: harnessruntime.Scope{SessionID: "session"},
	})

	tests := []struct {
		name    string
		handler string
		ctx     context.Context
		input   map[string]any
	}{
		{name: "read without runtime", handler: ToolReadMemory, ctx: bare, input: map[string]any{"path": "main"}},
		{name: "read negative bound", handler: ToolReadMemory, ctx: runtimeContext, input: map[string]any{"path": "main", "limit": -1}},
		{name: "read excessive bound", handler: ToolReadMemory, ctx: runtimeContext, input: map[string]any{"path": "main", "limit": 9}},
		{name: "write too large", handler: ToolWriteMemory, ctx: runtimeContext, input: map[string]any{"path": "main", "content": "12345", "mode": "replace"}},
		{name: "write invalid mode", handler: ToolWriteMemory, ctx: runtimeContext, input: map[string]any{"path": "main", "content": "x", "mode": "merge"}},
		{name: "write missing session", handler: ToolWriteMemory, ctx: harnessruntime.WithContext(bare, harnessruntime.ToolRuntime{}), input: map[string]any{"path": "main", "content": "x", "mode": "replace"}},
		{name: "delete missing version", handler: ToolDeleteMemory, ctx: runtimeContext, input: map[string]any{"path": "main"}},
		{name: "search negative matches", handler: ToolSearchMemory, ctx: runtimeContext, input: map[string]any{"query": "x", "limit": -1}},
		{name: "search excessive bytes", handler: ToolSearchMemory, ctx: runtimeContext, input: map[string]any{"query": "x", "max_bytes": 9}},
	}
	for _, current := range tests {
		t.Run(current.name, func(t *testing.T) {
			if err := executeMemoryHandler(t, handlers[current.handler], current.ctx, "call", current.handler, current.input); err == nil {
				t.Fatal("invalid memory operation succeeded")
			}
		})
	}

	// The mutation path separately requires the root tool-call identity.
	if _, err := handlers[ToolWriteMemory].Execute(runtimeContext, map[string]any{
		"path": "main", "content": "x", "mode": "replace",
	}, nil); err == nil || !strings.Contains(err.Error(), "tool-call ID") {
		t.Fatalf("missing call identity = %v", err)
	}

	backend.err = errors.New("backend failed")
	for _, current := range []struct {
		name  string
		input map[string]any
	}{
		{ToolReadMemory, map[string]any{"path": "main"}},
		{ToolWriteMemory, map[string]any{"path": "new", "content": "x", "mode": "replace"}},
		{ToolDeleteMemory, map[string]any{"path": "main", "expected_version": "v1"}},
		{ToolSearchMemory, map[string]any{"query": "x"}},
	} {
		if err := executeMemoryHandler(t, handlers[current.name], runtimeContext, "backend-"+current.name, current.name, current.input); err == nil {
			t.Fatalf("%s backend error was hidden", current.name)
		}
	}
}

func TestCapabilityScopeSearchAndRegistrationErrors(t *testing.T) {
	backend := &fakeStore{}
	store := storeOnly{store: backend}
	resolver := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "scope", nil })
	if _, err := New(Config{Store: store, Scope: resolver}); err == nil {
		t.Fatal("store without searcher succeeded")
	}
	if _, err := New(Config{Store: store, Searcher: backend, Scope: resolver}); err != nil {
		t.Fatalf("explicit searcher = %v", err)
	}
	for name, config := range map[string]Config{
		"invalid injection path":  {Store: backend, Scope: resolver, Injection: Injection{Enabled: true, MainPath: "../bad"}},
		"invalid injection bytes": {Store: backend, Scope: resolver, Injection: Injection{Enabled: true, MaxMainBytes: -1}},
		"invalid injection files": {Store: backend, Scope: resolver, Injection: Injection{Enabled: true, MaxFiles: -1}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := New(config); err == nil {
				t.Fatal("invalid configuration succeeded")
			}
		})
	}

	scopeError := errors.New("scope failed")
	for name, scope := range map[string]ScopeResolver{
		"resolver error": ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "", scopeError }),
		"invalid scope":  ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return " bad ", nil }),
	} {
		t.Run(name, func(t *testing.T) {
			handlers := capabilityHandlers(t, Config{Store: backend, Scope: scope})
			ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{})
			err := executeMemoryHandler(t, handlers[ToolReadMemory], ctx, "read", ToolReadMemory, map[string]any{"path": "main"})
			if err == nil {
				t.Fatal("invalid scope succeeded")
			}
		})
	}

	memoryCapability, _ := New(Config{Store: backend, Scope: resolver})
	conflictingTool, conflictingHandler := agentic.MustToolWithContext(
		ToolReadMemory, "conflict", func(context.Context, struct{}) (struct{}, error) { return struct{}{}, nil },
	)
	conflict := capability.Func{Name: "conflict", Apply: func(registry *capability.Registry) error {
		return registry.AddToolset(agentic.NewToolset().Add(conflictingTool, conflictingHandler))
	}}
	if _, err := capability.Compile(conflict, memoryCapability); !errors.Is(err, capability.ErrDuplicateTool) {
		t.Fatalf("duplicate tool registration = %v", err)
	}
	effectConflict := capability.Func{Name: "effect", Apply: func(registry *capability.Registry) error {
		return registry.AddEffectResolver(ToolReadMemory, memoryEffect("existing"))
	}}
	if _, err := capability.Compile(effectConflict, memoryCapability); err == nil {
		t.Fatal("duplicate effect registration succeeded")
	}

	failingScope := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) {
		return "", errors.New("scope failed")
	})
	handlers := capabilityHandlers(t, Config{Store: backend, Scope: failingScope})
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{SessionID: "session"})
	for _, current := range []struct {
		name  string
		input map[string]any
	}{
		{ToolWriteMemory, map[string]any{"path": "main", "content": "x", "mode": "replace"}},
		{ToolDeleteMemory, map[string]any{"path": "main", "expected_version": "v1"}},
		{ToolSearchMemory, map[string]any{"query": "x"}},
	} {
		if err := executeMemoryHandler(t, handlers[current.name], ctx, current.name, current.name, current.input); err == nil {
			t.Fatalf("%s scope error was hidden", current.name)
		}
	}
}

func TestMemoryValidatorsEffectsAndEmptyInjection(t *testing.T) {
	for _, scope := range []Scope{"", " bad", "bad\n", Scope(strings.Repeat("x", 257))} {
		if ValidateScope(scope) == nil {
			t.Fatalf("invalid scope %q succeeded", scope)
		}
	}
	for _, path := range []string{"trailing/", "a/.", "a/..", "a/" + strings.Repeat("x", 1024)} {
		if ValidatePath(path) == nil {
			t.Fatalf("invalid path %q succeeded", path)
		}
	}
	if ValidateList(ListOptions{Prefix: "../bad", Limit: 1}) == nil {
		t.Fatal("invalid list prefix succeeded")
	}
	for _, mutation := range []Mutation{
		{Path: "main", Kind: "unknown", IdempotencyKey: "id", Fingerprint: "fp"},
		{Path: "main", Kind: MutationDelete, Content: []byte("x"), IdempotencyKey: "id", Fingerprint: "fp"},
		{Path: "main", Kind: MutationReplace},
	} {
		if ValidateMutation(mutation) == nil {
			t.Fatalf("invalid mutation %#v succeeded", mutation)
		}
	}
	if value, err := bounded(1, 2, "test"); err != nil || value != 1 {
		t.Fatalf("bounded value = %d, %v", value, err)
	}
	if value, err := bounded(0, 2, "test"); err != nil || value != 2 {
		t.Fatalf("default bound = %d, %v", value, err)
	}
	if _, err := memoryEffect("read").ResolveEffect(context.Background(), agentic.ToolUse{Input: map[string]any{"path": "../bad"}}, nil); err == nil {
		t.Fatal("invalid effect path succeeded")
	}
	effect, err := memoryEffect("search").ResolveEffect(context.Background(), agentic.ToolUse{}, nil)
	if err != nil || effect.Resource.ID != "search" {
		t.Fatalf("search effect = %#v, %v", effect, err)
	}

	backend := &fakeStore{paths: []string{"notes/a"}}
	resolver := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "scope", nil })
	memoryCapability, _ := New(Config{Store: backend, Scope: resolver, Injection: Injection{Enabled: true}})
	plan, err := capability.Compile(memoryCapability)
	if err != nil {
		t.Fatal(err)
	}
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{})
	projection, err := plan.ContextPolicy().Project(ctx, contextpolicyRequest())
	if err != nil || len(projection.Messages) != 2 || !strings.Contains(projection.Messages[1].GetTextContent(), "notes/a") {
		t.Fatalf("empty-main injection = %#v, %v", projection, err)
	}
}

func TestInjectionReturnsListFailureAfterMissingMain(t *testing.T) {
	backend := &listFailureStore{fakeStore: &fakeStore{}, listErr: errors.New("list failed")}
	resolver := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "scope", nil })
	value, err := New(Config{Store: backend, Scope: resolver, Injection: Injection{Enabled: true}})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{})
	if _, err := plan.ContextPolicy().Project(ctx, contextpolicyRequest()); err == nil || !strings.Contains(err.Error(), "list failed") {
		t.Fatalf("list failure = %v", err)
	}
}
