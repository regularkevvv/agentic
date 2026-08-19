package standard

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/permission"
	providertest "github.com/regularkevvv/agentic/provider/test"

	uit "github.com/regularkevvv/agentic/tui"
	appconfig "github.com/regularkevvv/agentic/tui/config"
)

func TestRegistryAndFactoryValidation(t *testing.T) {
	t.Parallel()
	if _, err := NewRegistry(nil); err == nil {
		t.Fatal("nil factory succeeded")
	}
	missing := FactoryFunc{Name: "missing"}
	if _, err := missing.New(context.Background(), ModelConfig{}); err == nil {
		t.Fatal("factory without constructor succeeded")
	}
	factory := FactoryFunc{Name: "fake", Create: func(context.Context, ModelConfig) (agentic.Model, error) {
		return providertest.NewTestModel(providertest.ModelResponse{Text: "ok"}), nil
	}}
	registry, err := NewRegistry(factory)
	if err != nil {
		t.Fatal(err)
	}
	if err := registry.Register(factory); err == nil {
		t.Fatal("duplicate factory succeeded")
	}
	if err := (*Registry)(nil).Register(factory); err == nil {
		t.Fatal("nil registry registration succeeded")
	}
	if got, ok := registry.Get("fake"); !ok || got.ID() != "fake" {
		t.Fatalf("get = %#v, %v", got, ok)
	}
	if _, ok := (*Registry)(nil).Get("fake"); ok || (*Registry)(nil).IDs() != nil || !reflect.DeepEqual(registry.IDs(), []string{"fake"}) {
		t.Fatal("registry lookup/listing incorrect")
	}
}

func TestStandardToolPresentationCanBeOverridden(t *testing.T) {
	t.Parallel()
	defaults := resolveToolPresenter(nil)
	cases := []struct {
		name     string
		category uit.ToolCategory
	}{
		{"read_file", uit.ToolCategoryExplore},
		{"list_files", uit.ToolCategoryExplore},
		{"stat_file", uit.ToolCategoryExplore},
		{"read_artifact", uit.ToolCategoryExplore},
		{"run_command", uit.ToolCategoryExecute},
		{"write_file", uit.ToolCategoryChange},
		{"make_directory", uit.ToolCategoryChange},
		{"remove_path", uit.ToolCategoryChange},
		{"custom_tool", uit.ToolCategoryOther},
	}
	for _, test := range cases {
		got := defaults.PresentTool(uit.Tool{Name: test.name, Summary: "safe"})
		if got.Category != test.category || !strings.Contains(got.Title, "safe") || got.Detail != "" {
			t.Fatalf("%s presentation = %#v", test.name, got)
		}
	}
	if got := defaults.PresentTool(uit.Tool{Name: "run_command"}); got.Title != "Run command" {
		t.Fatalf("command fallback = %#v", got)
	}
	if got := defaults.PresentTool(uit.Tool{Name: "delegate_task", Summary: "secret task"}); got.Category != uit.ToolCategoryExplore || got.Title != "Delegate task" || strings.Contains(got.Title, "secret") {
		t.Fatalf("delegation presentation = %#v", got)
	}
	custom := uit.ToolPresenterFunc(func(uit.Tool) uit.ToolPresentation {
		return uit.ToolPresentation{Title: "Custom"}
	})
	if got := resolveToolPresenter(custom).PresentTool(uit.Tool{}); got.Title != "Custom" {
		t.Fatalf("custom presentation = %#v", got)
	}
}

func buildFixture(t *testing.T, factory ProviderFactory) (*Registry, Config) {
	t.Helper()
	base := t.TempDir()
	workspace := filepath.Join(base, "workspace")
	if err := os.Mkdir(workspace, 0o700); err != nil {
		t.Fatal(err)
	}
	prompt := filepath.Join(base, "prompt.md")
	if err := os.WriteFile(prompt, []byte("You are helpful."), 0o600); err != nil {
		t.Fatal(err)
	}
	registry, err := NewRegistry(factory)
	if err != nil {
		t.Fatal(err)
	}
	return registry, Config{
		ProfileName: "work", Provider: factory.ID(), Model: "model", ContextWindowTokens: 200000,
		SystemPromptFile: prompt, WorkspaceRoot: workspace, SessionDirectory: filepath.Join(base, "sessions"),
		Permission: "read-only",
	}
}

func TestBuildCreatesWorkingHarnessAdapter(t *testing.T) {
	t.Parallel()
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("standard assembly requires a supported OS sandbox backend")
	}
	var fakeModel *providertest.TestModel
	factory := FactoryFunc{Name: "fake", Create: func(_ context.Context, config ModelConfig) (agentic.Model, error) {
		if config.Model != "model" || config.ContextWindowTokens != 200000 {
			return nil, errors.New("bad config")
		}
		fakeModel = providertest.NewTestModel(providertest.ModelResponse{Text: "done"})
		return fakeModel, nil
	}}
	registry, config := buildFixture(t, factory)
	assembly, err := Build(context.Background(), registry, config)
	if err != nil {
		t.Fatal(err)
	}
	if assembly.Host == nil || assembly.Runtime == nil {
		t.Fatalf("assembly = %#v", assembly)
	}
	session, err := assembly.Host.NewSession(context.Background(), uit.SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Submit(context.Background(), uit.Input{Text: "task"}); err != nil {
		t.Fatal(err)
	}
	snapshot, _ := session.Snapshot(context.Background())
	if snapshot.ProfileLabel != "work" || snapshot.Workspace != config.WorkspaceRoot || (snapshot.Execution != "sandbox seatbelt" && snapshot.Execution != "sandbox landlock+seccomp") || len(snapshot.Transcript) < 2 {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	foundDelegation := false
	for _, tool := range fakeModel.Calls()[0].Tools {
		foundDelegation = foundDelegation || tool.Function.Name == "delegate_task"
	}
	if !foundDelegation {
		t.Fatal("standard model request omitted delegate_task")
	}
	_ = session.Close(context.Background())
	resolved := appconfig.Resolved{ProfileName: "p", Provider: "fake", Model: "m", ContextWindowTokens: 1, SystemPromptFile: "s", WorkspaceRoot: "w", SessionDirectory: "d", Permission: "workspace-write"}
	if got := FromResolved(resolved); got.ProfileName != "p" || got.PromptCacheRetention != agentic.PromptCacheShort {
		t.Fatalf("from resolved = %#v", got)
	}
}

func TestBuildFailsBeforeOrAtOwnedBoundary(t *testing.T) {
	t.Parallel()
	want := errors.New("credential")
	factory := FactoryFunc{Name: "fake", Create: func(context.Context, ModelConfig) (agentic.Model, error) { return nil, want }}
	registry, valid := buildFixture(t, factory)
	if _, err := Build(context.Background(), nil, valid); err == nil {
		t.Fatal("nil registry succeeded")
	}
	invalids := []Config{
		{},
		{Provider: "fake", Model: "m", ContextWindowTokens: 1},
		{Provider: "fake", Model: "m", ContextWindowTokens: 1, SystemPromptFile: "relative", WorkspaceRoot: "relative", SessionDirectory: "relative"},
	}
	for index, config := range invalids {
		if _, err := Build(context.Background(), registry, config); err == nil {
			t.Fatalf("invalid config %d succeeded", index)
		}
	}
	missingWorkspace := valid
	missingWorkspace.WorkspaceRoot = filepath.Join(t.TempDir(), "missing")
	if _, err := Build(context.Background(), registry, missingWorkspace); err == nil {
		t.Fatal("missing workspace succeeded")
	}
	fileWorkspace := valid
	fileWorkspace.WorkspaceRoot = valid.SystemPromptFile
	if _, err := Build(context.Background(), registry, fileWorkspace); err == nil {
		t.Fatal("file workspace succeeded")
	}
	missingPrompt := valid
	missingPrompt.SystemPromptFile = filepath.Join(t.TempDir(), "missing")
	if _, err := Build(context.Background(), registry, missingPrompt); err == nil {
		t.Fatal("missing prompt succeeded")
	}
	emptyPrompt := valid
	emptyPrompt.SystemPromptFile = filepath.Join(t.TempDir(), "empty")
	_ = os.WriteFile(emptyPrompt.SystemPromptFile, nil, 0o600)
	if _, err := Build(context.Background(), registry, emptyPrompt); err == nil {
		t.Fatal("empty prompt succeeded")
	}
	unregistered := valid
	unregistered.Provider = "other"
	if _, err := Build(context.Background(), registry, unregistered); err == nil {
		t.Fatal("unregistered provider succeeded")
	}
	if _, err := Build(context.Background(), registry, valid); !errors.Is(err, want) {
		t.Fatalf("factory error = %v", err)
	}
	nilFactory := FactoryFunc{Name: "nil", Create: func(context.Context, ModelConfig) (agentic.Model, error) { return nil, nil }}
	nilRegistry, nilConfig := buildFixture(t, nilFactory)
	if _, err := Build(context.Background(), nilRegistry, nilConfig); err == nil {
		t.Fatal("nil model succeeded")
	}
	successFactory := FactoryFunc{Name: "success", Create: func(context.Context, ModelConfig) (agentic.Model, error) {
		return providertest.NewTestModel(providertest.ModelResponse{Text: "ok"}), nil
	}}
	successRegistry, successConfig := buildFixture(t, successFactory)
	successConfig.Permission = "custom"
	if _, err := Build(context.Background(), successRegistry, successConfig); err == nil {
		t.Fatal("custom permission without policy succeeded")
	}
	successConfig.Permission = "bad"
	if _, err := Build(context.Background(), successRegistry, successConfig); err == nil {
		t.Fatal("bad permission succeeded")
	}
	successConfig.Permission = "workspace-write"
	successConfig.SessionDirectory = filepath.Join(successConfig.WorkspaceRoot, "sessions")
	if _, err := Build(context.Background(), successRegistry, successConfig); err == nil {
		t.Fatal("overlapping session directory succeeded")
	}
}

func TestPermissionsAndBuiltinCredentials(t *testing.T) {
	t.Parallel()
	if _, err := resolvePermission("bad", nil); err == nil {
		t.Fatal("bad permission succeeded")
	}
	if _, err := resolvePermission("custom", nil); err == nil {
		t.Fatal("custom without policy succeeded")
	}
	custom := permission.WorkspaceWrite()
	for _, test := range []struct {
		name   string
		custom *permission.Policy
	}{
		{"", nil}, {"workspace-write", nil}, {"read-only", nil}, {"custom", custom},
	} {
		if _, err := resolvePermission(test.name, test.custom); err != nil {
			t.Fatalf("permission %q: %v", test.name, err)
		}
	}
	missing := BuiltinFactories(func(string) (string, bool) { return "", false })
	if len(missing) != 3 {
		t.Fatalf("factories = %d", len(missing))
	}
	for _, factory := range missing {
		if _, err := factory.New(context.Background(), ModelConfig{Model: "m"}); err == nil || !strings.Contains(err.Error(), "credential") {
			t.Fatalf("%s credential error = %v", factory.ID(), err)
		}
	}
	available := BuiltinFactories(func(string) (string, bool) { return "key", true })
	for _, factory := range available {
		if model, err := factory.New(context.Background(), ModelConfig{Model: "m"}); err != nil || model == nil {
			t.Fatalf("%s = %#v, %v", factory.ID(), model, err)
		}
	}
	if len(BuiltinFactories(nil)) != 3 {
		t.Fatal("default environment lookup factories missing")
	}
}
