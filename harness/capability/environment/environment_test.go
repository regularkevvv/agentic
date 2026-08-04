package environment

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func TestEnvironmentCapabilityToolsUseInjectedLease(t *testing.T) {
	t.Parallel()
	var commands []env.Command
	environment, err := envmemory.New("/", func(_ context.Context, command env.Command) (env.CommandResult, error) {
		commands = append(commands, command)
		return env.CommandResult{Stdout: []byte("ok"), ExitCode: 3}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = environment.Close(context.Background()) }()
	value, err := New(Config{Files: true, Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	handlers := handlersByName(t, plan)
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{
		Environment: environment,
		SessionID:   "session",
	})
	if _, err := handlers[ToolMakeDirectory].Execute(ctx, map[string]any{"path": "/dir"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := handlers[ToolWriteFile].Execute(ctx, map[string]any{"path": "/dir/file", "content": "hello"}, nil); err != nil {
		t.Fatal(err)
	}
	read, err := handlers[ToolReadFile].Execute(ctx, map[string]any{"path": "/dir/file"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	readJSON, _ := json.Marshal(read)
	var decoded readFileOutput
	if err := json.Unmarshal(readJSON, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Content != "hello" || decoded.Bytes != 5 {
		t.Fatalf("read = %#v", decoded)
	}
	listed, err := handlers[ToolListFiles].Execute(ctx, map[string]any{"path": "/dir"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if entries := listed.([]env.DirEntry); len(entries) != 1 || entries[0].Name != "file" {
		t.Fatalf("entries = %#v", listed)
	}
	if _, err := handlers[ToolStatFile].Execute(ctx, map[string]any{"path": "/dir/file"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := handlers[ToolRunCommand].Execute(ctx, map[string]any{
		"name": "echo",
		"args": []any{"hello"},
	}, nil); err != nil {
		t.Fatal(err)
	}
	if len(commands) != 1 || commands[0].Name != "echo" ||
		!reflect.DeepEqual(commands[0].Args, []string{"hello"}) {
		t.Fatalf("commands = %#v", commands)
	}
	if _, err := handlers[ToolRemovePath].Execute(ctx, map[string]any{"path": "/dir/file"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := environment.ReadFile(ctx, "/dir/file"); !env.HasCode(err, env.CodeNotFound) {
		t.Fatalf("removed read error = %v", err)
	}
}

func TestEnvironmentCapabilityValidation(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{}); err == nil {
		t.Fatal("empty capability succeeded")
	}
	value, err := New(Config{Files: true})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(value)
	if err != nil {
		t.Fatal(err)
	}
	handlers := handlersByName(t, plan)
	if _, err := handlers[ToolWriteFile].Execute(context.Background(), map[string]any{
		"path": "file", "content": "x",
	}, nil); err == nil {
		t.Fatal("tool without runtime succeeded")
	}
	environment, _ := envmemory.New("/", nil)
	defer func() { _ = environment.Close(context.Background()) }()
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{Environment: environment})
	if _, err := handlers[ToolWriteFile].Execute(ctx, map[string]any{
		"path": "file", "content": "x", "mode": float64(0o1000),
	}, nil); err == nil {
		t.Fatal("invalid mode succeeded")
	}
}

func handlersByName(t *testing.T, plan capability.Plan) map[string]agentic.ToolHandler {
	t.Helper()
	result := make(map[string]agentic.ToolHandler)
	for _, toolset := range plan.Toolsets() {
		tools, handlers := toolset.ToolsAndHandlers()
		if len(tools) != len(handlers) {
			t.Fatal(errors.New("toolset cardinality mismatch"))
		}
		for index, tool := range tools {
			result[tool.Function.Name] = handlers[index]
		}
	}
	return result
}

func TestEnvironmentToolFailuresModesAndShellOutput(t *testing.T) {
	t.Parallel()
	shellOnly, err := New(Config{Shell: true})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := capability.Compile(shellOnly)
	if err != nil {
		t.Fatal(err)
	}
	if handlers := handlersByName(t, plan); len(handlers) != 1 || handlers[ToolRunCommand] == nil {
		t.Fatalf("shell-only handlers = %#v", handlers)
	}

	files, _ := New(Config{Files: true})
	plan, err = capability.Compile(files)
	if err != nil {
		t.Fatal(err)
	}
	handlers := handlersByName(t, plan)
	environment, err := envmemory.New("/", nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{Environment: environment})
	if _, err := handlers[ToolMakeDirectory].Execute(ctx, map[string]any{
		"path": "/bad", "mode": float64(0o1000),
	}, nil); err == nil {
		t.Fatal("invalid directory mode succeeded")
	}
	if _, err := handlers[ToolWriteFile].Execute(ctx, map[string]any{
		"path": "/default-mode", "content": "x",
	}, nil); err != nil {
		t.Fatal(err)
	}
	if err := environment.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]map[string]any{
		ToolReadFile:      {"path": "/default-mode"},
		ToolWriteFile:     {"path": "/closed", "content": "x"},
		ToolListFiles:     {"path": "/"},
		ToolStatFile:      {"path": "/"},
		ToolMakeDirectory: {"path": "/closed"},
		ToolRemovePath:    {"path": "/default-mode"},
	} {
		if _, err := handlers[name].Execute(ctx, input, nil); err == nil {
			t.Fatalf("%s on closed environment succeeded", name)
		}
	}

	noShell, _ := envmemory.New("/", nil)
	defer func() { _ = noShell.Close(context.Background()) }()
	all, _ := New(Config{Files: true, Shell: true})
	plan, _ = capability.Compile(all)
	run := handlersByName(t, plan)[ToolRunCommand]
	noShellCtx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{Environment: noShell})
	if _, err := run.Execute(noShellCtx, map[string]any{"name": ""}, nil); err == nil {
		t.Fatal("empty command succeeded")
	}
	if _, err := run.Execute(noShellCtx, map[string]any{"name": "echo"}, nil); !env.HasCode(err, env.CodeUnsupported) {
		t.Fatalf("missing shell error = %v", err)
	}

	var got env.Command
	shell, _ := envmemory.New("/", func(_ context.Context, command env.Command) (env.CommandResult, error) {
		got = command
		return env.CommandResult{
			Stdout:   []byte{'o', 0xff},
			Stderr:   []byte{'e', 0xff},
			ExitCode: 9,
		}, errors.New("command warning")
	})
	defer func() { _ = shell.Close(context.Background()) }()
	shellCtx := harnessruntime.WithContext(context.Background(), harnessruntime.ToolRuntime{Environment: shell})
	output, err := run.Execute(shellCtx, map[string]any{
		"name":  "tool",
		"args":  []any{"one"},
		"dir":   "/work",
		"env":   []any{"A=B"},
		"stdin": "input",
	}, nil)
	if err == nil {
		t.Fatal("shell error was hidden")
	}
	encoded, _ := json.Marshal(output)
	var decoded commandOutput
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(decoded.Stdout, "�") || !strings.Contains(decoded.Stderr, "�") ||
		decoded.ExitCode != 9 || got.Dir != "/work" || string(got.Stdin) != "input" ||
		!reflect.DeepEqual(got.Env, []string{"A=B"}) {
		t.Fatalf("output=%#v command=%#v", decoded, got)
	}
}

func TestEnvironmentEffectResolversCanonicalizeInputs(t *testing.T) {
	t.Parallel()
	environment, err := envmemory.New("/workspace", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = environment.Close(context.Background()) }()
	resolver := fileEffect("read")
	if _, err := resolver.ResolveEffect(context.Background(), agentic.ToolUse{
		ID: "bad", Name: ToolReadFile,
	}, environment); err == nil {
		t.Fatal("missing path effect succeeded")
	}
	effect, err := resolver.ResolveEffect(context.Background(), agentic.ToolUse{
		ID: "read", Name: ToolReadFile, Input: map[string]any{"path": "file"},
	}, environment)
	if err != nil || effect.Capability != "filesystem" || effect.Action != "read" ||
		!effect.Resource.Valid() {
		t.Fatalf("file effect = %#v, %v", effect, err)
	}
	if _, err := commandEffect(context.Background(), agentic.ToolUse{}, environment); err == nil {
		t.Fatal("missing command effect succeeded")
	}
	effect, err = commandEffect(context.Background(), agentic.ToolUse{
		Input: map[string]any{"name": "echo", "args": []any{"hello"}, "dir": "/workspace"},
	}, environment)
	if err != nil || effect.Capability != "shell" || effect.Action != "exec" ||
		effect.Resource.Scheme != "command" || effect.Resource.Display != "echo hello · in /workspace" {
		t.Fatalf("command effect = %#v, %v", effect, err)
	}
	if _, err := commandEffect(context.Background(), agentic.ToolUse{
		Input: map[string]any{"name": "echo", "args": make(chan int)},
	}, environment); err == nil {
		t.Fatal("unencodable command effect succeeded")
	}
	if mode, err := fileMode(0o600, 0o644); err != nil || mode != 0o600 {
		t.Fatalf("explicit file mode = %v, %v", mode, err)
	}
}

func TestToolSummaryShowsUsefulArgumentsWithoutSecrets(t *testing.T) {
	t.Parallel()
	call := agentic.ToolUse{Name: ToolRunCommand, Input: map[string]any{
		"name": "curl",
		"args": []any{
			"-H", "Authorization: Bearer header-secret",
			"--api-key", "flag-secret",
			"--password=inline-secret",
			"https://example.test/?token=query-secret&ok=yes",
			"two words",
		},
		"dir": "/work tree", "env": []any{"API_KEY=env-secret"}, "stdin": "stdin-secret",
	}}
	got := ToolSummary(call)
	for _, want := range []string{"curl", "-H", "Authorization", "--api-key", "--password=[redacted]", "token=[redacted]", `"two words"`, `· in "/work tree"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("summary lacks %q: %q", want, got)
		}
	}
	for _, secret := range []string{"header-secret", "flag-secret", "inline-secret", "query-secret", "env-secret", "stdin-secret"} {
		if strings.Contains(got, secret) {
			t.Fatalf("summary leaked %q: %q", secret, got)
		}
	}
	if path := ToolSummary(agentic.ToolUse{Name: ToolReadFile, Input: map[string]any{"path": "README.md"}}); path != "README.md" {
		t.Fatalf("path summary = %q", path)
	}
	if unknown := ToolSummary(agentic.ToolUse{Name: "custom", Input: map[string]any{"token": "secret"}}); unknown != "" {
		t.Fatalf("unknown summary = %q", unknown)
	}
}
