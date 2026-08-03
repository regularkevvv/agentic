package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/regularkevvv/agentic/tui/app"
	"github.com/regularkevvv/agentic/tui/render"
)

func writeConfig(t *testing.T, directory, name, value string) string {
	t.Helper()
	path := filepath.Join(directory, name)
	if err := os.WriteFile(path, []byte(value), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadMergesFilesAndResolveAppliesFlags(t *testing.T) {
	directory := t.TempDir()
	workspace := filepath.Join(directory, "workspace")
	if err := os.Mkdir(workspace, 0o700); err != nil {
		t.Fatal(err)
	}
	prompt := writeConfig(t, directory, "prompt.md", "system")
	resume := true
	_ = resume
	user := writeConfig(t, directory, "user.toml", `
[ui]
alternate_screen = "always"
color = "never"
thinking = "visible"
tool_details = "expanded"
preview_hz = 30
[session]
directory = "`+filepath.Join(directory, "sessions-user")+`"
resume_last = true
[profile.work]
provider = "anthropic"
model = "old"
context_window_tokens = 100
system_prompt_file = "`+prompt+`"
permission = "read-only"
[profile.work.workspace]
root = "`+workspace+`"
`)
	project := writeConfig(t, directory, "project.toml", `
[ui]
alternate_screen = "never"
[session]
directory = "`+filepath.Join(directory, "sessions-project")+`"
[profile.work]
model = "new"
context_window_tokens = 200
`)
	document, err := Load(filepath.Join(directory, "missing.toml"), user, project)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := Resolve(document, "work", Flags{
		Provider: "openai", Model: "flag-model", ContextWindowTokens: 300,
		SystemPromptFile: prompt, Permission: "read-only", WorkspaceRoot: workspace,
		SessionDirectory: filepath.Join(directory, "sessions-project"),
		AlternateScreen:  "auto", Color: "auto", Thinking: "hidden", ToolDetails: "collapsed", PreviewHz: 120,
	}, []string{"NO_COLOR=1", "TERM=xterm"})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Provider != "openai" || resolved.Model != "flag-model" || resolved.ContextWindowTokens != 300 || resolved.Permission != "read-only" || !resolved.ResumeLast {
		t.Fatalf("resolved = %#v", resolved)
	}
	if resolved.UI.AlternateScreen != app.AlternateAuto || !resolved.UI.NoColor || resolved.UI.Thinking != render.ThinkingHidden || resolved.UI.ToolDetails || resolved.UI.PreviewHz != 120 {
		t.Fatalf("UI = %#v", resolved.UI)
	}
	if resolved.SessionDirectory != filepath.Join(directory, "sessions-project") {
		t.Fatalf("session directory = %s", resolved.SessionDirectory)
	}
}

func TestResolveDefaultsDerivedDirectoryAndUI(t *testing.T) {
	directory := t.TempDir()
	prompt := writeConfig(t, directory, "prompt.md", "system")
	document := Document{Profiles: map[string]Profile{"work": {
		Provider: "openai", Model: "model", ContextWindowTokens: 100,
		SystemPromptFile: prompt, Workspace: Workspace{Root: directory},
	}}}
	resolved, err := Resolve(document, "work", Flags{}, []string{"TERM=xterm"})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Permission != "workspace-write" || !resolved.ResumeLast || resolved.UI.AlternateScreen != app.AlternateAuto || resolved.UI.NoColor || resolved.UI.PreviewHz != 60 || !strings.Contains(resolved.SessionDirectory, filepath.Join("agentic", "sessions")) {
		t.Fatalf("resolved = %#v", resolved)
	}
	if WorkspacePath(directory) != filepath.Join(directory, ".agentic.toml") {
		t.Fatal("workspace path incorrect")
	}
	if path, err := UserPath(); err != nil || !strings.HasSuffix(path, filepath.Join("agentic", "config.toml")) {
		t.Fatalf("user path = %q, %v", path, err)
	}
}

func TestLoadAndResolveErrors(t *testing.T) {
	directory := t.TempDir()
	malformed := writeConfig(t, directory, "bad.toml", "[broken")
	if _, err := Load(malformed); err == nil {
		t.Fatal("malformed config succeeded")
	}
	if _, err := Load(directory); err == nil {
		t.Fatal("directory config succeeded")
	}
	if _, err := Resolve(Document{}, "", Flags{}, nil); err == nil {
		t.Fatal("empty profile succeeded")
	}
	if _, err := Resolve(Document{}, "missing", Flags{}, nil); err == nil {
		t.Fatal("missing profile succeeded")
	}
	base := Profile{Provider: "p", Model: "m", ContextWindowTokens: 1, SystemPromptFile: "prompt", Workspace: Workspace{Root: directory}}
	cases := []Profile{
		{Model: "m", ContextWindowTokens: 1, SystemPromptFile: "p", Workspace: Workspace{Root: directory}},
		{Provider: "p", ContextWindowTokens: 1, SystemPromptFile: "p", Workspace: Workspace{Root: directory}},
		{Provider: "p", Model: "m", SystemPromptFile: "p", Workspace: Workspace{Root: directory}},
		{Provider: "p", Model: "m", ContextWindowTokens: 1, Workspace: Workspace{Root: directory}},
		{Provider: "p", Model: "m", ContextWindowTokens: 1, SystemPromptFile: "p"},
	}
	for index, profile := range cases {
		if _, err := Resolve(Document{Profiles: map[string]Profile{"x": profile}}, "x", Flags{}, nil); err == nil {
			t.Fatalf("missing-field case %d succeeded", index)
		}
	}
	for _, permission := range []string{"invalid", "custom"} {
		profile := base
		profile.Permission = permission
		_, err := Resolve(Document{Profiles: map[string]Profile{"x": profile}}, "x", Flags{}, nil)
		if permission == "invalid" && err == nil {
			t.Fatal("invalid permission succeeded")
		}
		if permission == "custom" && err != nil {
			t.Fatalf("custom permission config failed early: %v", err)
		}
	}
	for _, flags := range []Flags{{Color: "sepia"}, {ToolDetails: "raw"}, {AlternateScreen: "bad"}, {Thinking: "bad"}, {PreviewHz: 241}} {
		if _, err := Resolve(Document{Profiles: map[string]Profile{"x": base}}, "x", flags, nil); err == nil {
			t.Fatalf("invalid UI flags %#v succeeded", flags)
		}
	}
}

func TestMergeHelpersAndExpansion(t *testing.T) {
	target := Document{}
	resume := false
	merge(&target, Document{UI: UI{AlternateScreen: "always", Color: "always", Thinking: "visible", ToolDetails: "expanded", PreviewHz: 12}, Session: Session{Directory: "/tmp/s", ResumeLast: &resume}, Profiles: map[string]Profile{"x": {Provider: "p", Model: "m", ContextWindowTokens: 1, SystemPromptFile: "s", Permission: "custom", Workspace: Workspace{Root: "w"}}}})
	if target.Profiles["x"].Workspace.Root != "w" || target.Session.ResumeLast == nil || *target.Session.ResumeLast {
		t.Fatalf("merged = %#v", target)
	}
	if first("", "x", "y") != "x" || first() != "" || !hasEnvironment([]string{"A=1", "B"}, "A") || !hasEnvironment([]string{"B"}, "B") || hasEnvironment(nil, "A") {
		t.Fatal("small helpers incorrect")
	}
	if _, err := absoluteExpanded("."); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", t.TempDir())
	if expanded, err := absoluteExpanded("~/file"); err != nil || !strings.HasSuffix(expanded, "file") {
		t.Fatalf("home expansion = %q, %v", expanded, err)
	}
	for _, test := range []struct {
		file  UI
		flags Flags
		want  bool
	}{
		{flags: Flags{Color: "always"}, want: false},
		{flags: Flags{Color: "never"}, want: true},
	} {
		ui, err := resolveUI(test.file, test.flags, nil)
		if err != nil || ui.NoColor != test.want {
			t.Fatalf("resolved UI = %#v, %v", ui, err)
		}
	}
}
