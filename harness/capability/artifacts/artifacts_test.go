package artifacts

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/artifact"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type getFailureStore struct {
	err error
}

func (s getFailureStore) Put(context.Context, string, string, []byte) (artifact.Handle, error) {
	return "", s.err
}

func (s getFailureStore) Get(context.Context, string, artifact.Handle) ([]byte, error) {
	return nil, s.err
}

func TestReadArtifactIsBoundedUTF8AndSessionScoped(t *testing.T) {
	t.Parallel()
	store := artifactmemory.New()
	data := []byte(strings.Repeat("é", 20))
	handle, err := store.Put(context.Background(), "session_one", "call", data)
	if err != nil {
		t.Fatal(err)
	}
	value, err := New(Config{Store: store, MaxReadBytes: 9})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	var handler agentic.ToolHandler
	tools, handlers := plan.Toolsets()[0].ToolsAndHandlers()
	for index, tool := range tools {
		if tool.Function.Name == ToolReadArtifact {
			handler = handlers[index]
		}
	}
	if handler == nil {
		t.Fatal("read_artifact handler missing")
	}
	environment, _ := envmemory.New("/", nil)
	defer func() { _ = environment.Close(context.Background()) }()
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environment,
		SessionID:   "session_one",
	})
	first, err := handler.Execute(ctx, map[string]any{"handle": handle.String()}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		NextOffset int    `json:"next_offset"`
		TotalBytes int    `json:"total_bytes"`
		EOF        bool   `json:"eof"`
		Content    string `json:"content"`
	}
	payload, _ := json.Marshal(first)
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.NextOffset != 8 || decoded.TotalBytes != len(data) || decoded.EOF ||
		decoded.Content != strings.Repeat("é", 4) {
		t.Fatalf("first chunk = %#v", decoded)
	}
	if _, err := handler.Execute(ctx, map[string]any{"handle": handle.String(), "offset": float64(1)}, nil); err == nil {
		t.Fatal("non-UTF8 offset succeeded")
	}
	other := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environment,
		SessionID:   "session_two",
	})
	if _, err := handler.Execute(other, map[string]any{"handle": handle.String()}, nil); err == nil {
		t.Fatal("cross-session artifact read succeeded")
	}
}

func TestArtifactCapabilityValidation(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{}); err == nil {
		t.Fatal("nil store succeeded")
	}
	if _, err := New(Config{Store: artifactmemory.New(), MaxReadBytes: -1}); err == nil {
		t.Fatal("negative limit succeeded")
	}
	value, _ := New(Config{Store: artifactmemory.New()})
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	_, handlers := plan.Toolsets()[0].ToolsAndHandlers()
	if _, err := handlers[0].Execute(context.Background(), map[string]any{"handle": "bad"}, nil); err == nil {
		t.Fatal("tool without runtime succeeded")
	}
}

func TestReadArtifactValidationPagingAndEffect(t *testing.T) {
	t.Parallel()
	store := artifactmemory.New()
	handle, err := store.Put(context.Background(), "session", "call", []byte("abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	value, err := New(Config{Store: store, MaxReadBytes: 4})
	if err != nil {
		t.Fatal(err)
	}
	var resolver capability.EffectResolver
	plan, err := capability.Compile(
		value,
		capability.Func{Name: "inspect", Order: capability.Ordering{After: []string{ID}}, Apply: func(registry *capability.Registry) error {
			var ok bool
			resolver, ok = registry.EffectResolver(ToolReadArtifact)
			if !ok {
				return errors.New("artifact effect missing")
			}
			return nil
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, handlers := plan.Toolsets()[0].ToolsAndHandlers()
	handler := handlers[0]
	environment, _ := envmemory.New("/", nil)
	defer func() { _ = environment.Close(context.Background()) }()
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environment,
		SessionID:   "session",
	})
	for name, input := range map[string]map[string]any{
		"invalid handle": {"handle": "bad"},
		"negative offset": {
			"handle": handle.String(), "offset": float64(-1),
		},
		"negative limit": {
			"handle": handle.String(), "limit": float64(-1),
		},
		"past end": {
			"handle": handle.String(), "offset": float64(7),
		},
	} {
		if _, err := handler.Execute(ctx, input, nil); err == nil {
			t.Fatalf("%s succeeded", name)
		}
	}
	page, err := handler.Execute(ctx, map[string]any{
		"handle": handle.String(), "offset": float64(2), "limit": float64(100),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(page)
	var decoded struct {
		Handle     string `json:"handle"`
		Offset     int    `json:"offset"`
		NextOffset int    `json:"next_offset"`
		TotalBytes int    `json:"total_bytes"`
		EOF        bool   `json:"eof"`
		Content    string `json:"content"`
	}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Handle != handle.String() || decoded.Offset != 2 || decoded.NextOffset != 6 ||
		decoded.TotalBytes != 6 || !decoded.EOF || decoded.Content != "cdef" {
		t.Fatalf("page = %#v", decoded)
	}
	effect, err := resolver.ResolveEffect(context.Background(), agentic.ToolUse{
		Input: map[string]any{"handle": handle.String()},
	}, environment)
	if err != nil || effect.Capability != "artifact" || effect.Action != "read" ||
		effect.Resource != (env.CanonicalResource{Scheme: "artifact", ID: handle.String(), Display: handle.String()}) {
		t.Fatalf("artifact effect = %#v, %v", effect, err)
	}
	if _, err := resolver.ResolveEffect(context.Background(), agentic.ToolUse{
		Input: map[string]any{"handle": "bad"},
	}, environment); err == nil {
		t.Fatal("invalid artifact effect succeeded")
	}
}

func TestReadArtifactPropagatesStoreFailure(t *testing.T) {
	boom := errors.New("store failed")
	value, err := New(Config{Store: getFailureStore{err: boom}})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	_, handlers := plan.Toolsets()[0].ToolsAndHandlers()
	environment, err := envmemory.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = environment.Close(context.Background()) }()
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environment,
		SessionID:   "session",
	})
	handle := "art_" + strings.Repeat("0", 64)
	if _, err := handlers[0].Execute(ctx, map[string]any{"handle": handle}, nil); !errors.Is(err, boom) {
		t.Fatalf("store failure = %v", err)
	}
}

func TestReadArtifactShortPageAndFrozenRegistration(t *testing.T) {
	store := artifactmemory.New()
	handle, err := store.Put(context.Background(), "session", "short", []byte("abc"))
	if err != nil {
		t.Fatal(err)
	}
	value, err := New(Config{Store: store, MaxReadBytes: 4})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	_, handlers := plan.Toolsets()[0].ToolsAndHandlers()
	environment, err := envmemory.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = environment.Close(context.Background()) }()
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environment,
		SessionID:   "session",
	})
	if _, err := handlers[0].Execute(ctx, map[string]any{"handle": handle.String()}, nil); err != nil {
		t.Fatalf("short page = %v", err)
	}

	var frozen *capability.Registry
	if _, err := capability.Compile(capability.Func{
		Name: "capture",
		Apply: func(registry *capability.Registry) error {
			frozen = registry
			return nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := value.Register(frozen); !errors.Is(err, capability.ErrRegistryFrozen) {
		t.Fatalf("frozen artifact registration = %v", err)
	}
}
