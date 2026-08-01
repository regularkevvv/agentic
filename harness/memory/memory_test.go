package memory

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type fakeStore struct {
	mu        sync.Mutex
	file      File
	paths     []string
	mutations []Mutation
	search    SearchResult
	err       error
	scopes    []Scope
}

func (s *fakeStore) Read(_ context.Context, scope Scope, path string, options ReadOptions) (File, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scopes = append(s.scopes, scope)
	if s.err != nil {
		return File{}, s.err
	}
	if s.file.Path == "" || s.file.Path != path {
		return File{}, ErrNotFound
	}
	if len(s.file.Content) > options.MaxBytes {
		return File{}, ErrLimitExceeded
	}
	return CloneFile(s.file), nil
}

func (s *fakeStore) List(_ context.Context, scope Scope, options ListOptions) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scopes = append(s.scopes, scope)
	if s.err != nil {
		return nil, s.err
	}
	values := append([]string(nil), s.paths...)
	if len(values) > options.Limit {
		values = values[:options.Limit]
	}
	return values, nil
}

func (s *fakeStore) Mutate(_ context.Context, scope Scope, mutation Mutation) (MutationResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scopes = append(s.scopes, scope)
	s.mutations = append(s.mutations, mutation)
	if s.err != nil {
		return MutationResult{}, s.err
	}
	return MutationResult{Path: mutation.Path, Version: "next", Deleted: mutation.Kind == MutationDelete, Bytes: len(mutation.Content)}, nil
}

func (s *fakeStore) Search(_ context.Context, scope Scope, _ SearchOptions) (SearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.scopes = append(s.scopes, scope)
	return s.search, s.err
}

func TestCapabilityToolsDeriveScopeAndIdempotency(t *testing.T) {
	store := &fakeStore{
		file:   File{Path: "main", Content: []byte("remember"), Version: "v1"},
		paths:  []string{"main"},
		search: SearchResult{Matches: []Match{{Path: "main", Content: "remember"}}},
	}
	resolved := harnessruntime.Scope{}
	memoryCapability, err := New(Config{
		Store: store,
		Scope: ScopeResolverFunc(func(_ context.Context, scope harnessruntime.Scope) (Scope, error) {
			resolved = scope
			return "tenant", nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(memoryCapability)
	if err != nil {
		t.Fatal(err)
	}
	registry := agentic.NewRegistry()
	for _, toolset := range plan.Toolsets() {
		if err := agentic.RegisterToolset(registry, toolset); err != nil {
			t.Fatal(err)
		}
	}
	base := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		SessionID: "session", Scope: harnessruntime.Scope{SessionID: "session", Agent: "worker"},
	})
	read, _ := registry.Execute(agentic.WithToolCallContext(base, agentic.ToolCallContext{ID: "read", Name: ToolReadMemory, Attempt: 1}), agentic.ToolUse{
		ID: "read", Name: ToolReadMemory, Input: map[string]any{"path": "main"},
	}, nil)
	if read.IsError || !strings.Contains(agentic.FormatToolResult(read.Content), "remember") {
		t.Fatalf("read = %#v", read)
	}
	write, _ := registry.Execute(agentic.WithToolCallContext(base, agentic.ToolCallContext{ID: "write", Name: ToolWriteMemory, Attempt: 1}), agentic.ToolUse{
		ID: "write", Name: ToolWriteMemory,
		Input: map[string]any{"path": "main", "content": "new", "mode": "replace", "expected_version": "v1"},
	}, nil)
	if write.IsError {
		t.Fatalf("write = %#v", write)
	}
	deleted, _ := registry.Execute(agentic.WithToolCallContext(base, agentic.ToolCallContext{ID: "delete", Name: ToolDeleteMemory, Attempt: 1}), agentic.ToolUse{
		ID: "delete", Name: ToolDeleteMemory, Input: map[string]any{"path": "main", "expected_version": "v1"},
	}, nil)
	if deleted.IsError {
		t.Fatalf("delete = %#v", deleted)
	}
	searched, _ := registry.Execute(agentic.WithToolCallContext(base, agentic.ToolCallContext{ID: "search", Name: ToolSearchMemory, Attempt: 1}), agentic.ToolUse{
		ID: "search", Name: ToolSearchMemory, Input: map[string]any{"query": "remember"},
	}, nil)
	if searched.IsError || resolved.SessionID != "session" {
		t.Fatalf("search = %#v scope=%#v", searched, resolved)
	}
	store.mu.Lock()
	mutations := append([]Mutation(nil), store.mutations...)
	store.mu.Unlock()
	if len(mutations) != 2 || mutations[0].IdempotencyKey != "session/write" || !strings.HasPrefix(mutations[0].Fingerprint, "sha256:") ||
		mutations[1].IdempotencyKey != "session/delete" {
		t.Fatalf("mutations = %#v", mutations)
	}
	effect, err := memoryEffect("read").ResolveEffect(context.Background(), agentic.ToolUse{Input: map[string]any{"path": "main"}}, nil)
	if err != nil || effect.Resource.ID != "main" || effect.Action != "read" {
		t.Fatalf("effect = %#v, %v", effect, err)
	}
}

func TestEphemeralInjectionIsCurrentBoundedAndFailPolicyAware(t *testing.T) {
	store := &fakeStore{file: File{Path: "MEMORY.md", Content: []byte("first"), Version: "v1"}, paths: []string{"MEMORY.md", "notes/a"}}
	resolver := ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "tenant", nil })
	memoryCapability, _ := New(Config{
		Store: store, Scope: resolver,
		Injection: Injection{Enabled: true, MaxMainBytes: 20, MaxFiles: 2},
	})
	plan, err := capability.Compile(memoryCapability)
	if err != nil {
		t.Fatal(err)
	}
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{Scope: harnessruntime.Scope{SessionID: "session"}})
	project := func() []agentic.Message {
		result, err := plan.ContextPolicy().Project(ctx, contextpolicyRequest())
		if err != nil {
			t.Fatal(err)
		}
		if len(result.DurableAdditions) != 0 {
			t.Fatal("memory injection became durable")
		}
		return result.Messages
	}
	first := project()
	if len(first) != 2 || !strings.Contains(first[1].GetTextContent(), "first") || !strings.Contains(first[1].GetTextContent(), "untrusted") {
		t.Fatalf("first projection = %#v", first)
	}
	store.mu.Lock()
	store.file.Content = []byte("second")
	store.mu.Unlock()
	second := project()
	if !strings.Contains(second[1].GetTextContent(), "second") || strings.Contains(second[1].GetTextContent(), "first") {
		t.Fatalf("second projection = %#v", second)
	}

	store.err = errors.New("backend unavailable")
	failOpen, _ := New(Config{Store: store, Scope: resolver, Injection: Injection{Enabled: true, FailOpen: true}})
	openPlan, _ := capability.Compile(failOpen)
	result, err := openPlan.ContextPolicy().Project(ctx, contextpolicyRequest())
	if err != nil || len(result.Messages) != 1 {
		t.Fatalf("fail open = %#v, %v", result, err)
	}
	failClosed, _ := New(Config{Store: store, Scope: resolver, Injection: Injection{Enabled: true}})
	closedPlan, _ := capability.Compile(failClosed)
	if _, err := closedPlan.ContextPolicy().Project(ctx, contextpolicyRequest()); err == nil {
		t.Fatal("fail-closed injection succeeded")
	}
	scopeFailure, _ := New(Config{Store: store, Scope: ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) {
		return "", errors.New("scope unavailable")
	}), Injection: Injection{Enabled: true, FailOpen: true}})
	scopePlan, _ := capability.Compile(scopeFailure)
	if _, err := scopePlan.ContextPolicy().Project(ctx, contextpolicyRequest()); err == nil {
		t.Fatal("scope error failed open")
	}
}

func contextpolicyRequest() contextpolicy.ProjectionRequest {
	return contextpolicy.ProjectionRequest{Messages: []agentic.Message{agentic.NewTextMessage(agentic.RoleUser, "prompt")}}
}

func TestValidationAndCapabilityErrors(t *testing.T) {
	valid := Mutation{Path: "notes/main", Kind: MutationReplace, IdempotencyKey: "id", Fingerprint: "fp"}
	if ValidateMutation(valid) != nil || ValidateRead("notes/main", ReadOptions{MaxBytes: 1}) != nil ||
		ValidateList(ListOptions{Limit: 1}) != nil || ValidateSearch(SearchOptions{Query: "q", Limit: 1, MaxBytes: 1}) != nil {
		t.Fatal("valid memory values rejected")
	}
	for _, value := range []string{"", "/abs", "a/../b", "a\\b", "a//b", ".agentic-memory.jsonl"} {
		if ValidatePath(value) == nil {
			t.Fatalf("invalid path %q succeeded", value)
		}
	}
	if ValidateScope("") == nil || ValidateRead("main", ReadOptions{}) == nil || ValidateList(ListOptions{}) == nil ||
		ValidateSearch(SearchOptions{}) == nil || ValidateMutation(Mutation{}) == nil {
		t.Fatal("invalid bounded operation succeeded")
	}
	file := File{Content: []byte("x")}
	copy := CloneFile(file)
	copy.Content[0] = 'y'
	if reflect.DeepEqual(file, copy) {
		t.Fatal("file clone shares content")
	}
	if _, err := New(Config{}); err == nil {
		t.Fatal("missing memory dependencies succeeded")
	}
	store := &fakeStore{}
	if _, err := New(Config{Store: store, Scope: ScopeResolverFunc(func(context.Context, harnessruntime.Scope) (Scope, error) { return "scope", nil }), Limits: CapabilityLimits{MaxReadBytes: -1}}); err == nil {
		t.Fatal("negative capability limits succeeded")
	}
}
