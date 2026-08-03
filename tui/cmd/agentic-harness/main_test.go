package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFlags(t *testing.T) {
	t.Parallel()
	values, err := parseFlags([]string{
		"--offline", "--no-alt-screen", "--no-color", "--profile", "p",
		"--config", "c", "--workspace-config", "w", "--resume", "s",
		"--provider", "openai", "--model", "m", "--context-window", "100",
		"--system-prompt-file", "prompt", "--permission", "read-only",
		"--workspace", "/w", "--session-dir", "/s", "--preview-hz", "30",
		"--thinking", "hidden", "--tool-details", "expanded", "--color", "never",
		"--alternate-screen", "always",
	}, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if !values.offline || !values.noAltScreen || !values.noColor || values.profile != "p" || values.contextWindow != 100 || values.previewHz != 30 || values.alternateScreen != "always" {
		t.Fatalf("flags = %#v", values)
	}
	if _, err := parseFlags([]string{"positional"}, &bytes.Buffer{}); err == nil {
		t.Fatal("positional argument succeeded")
	}
	if _, err := parseFlags([]string{"--help"}, &bytes.Buffer{}); !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("help error = %v", err)
	}
}

func TestRunRejectsBeforeTerminalStartup(t *testing.T) {
	t.Parallel()
	var output bytes.Buffer
	if err := run(context.Background(), []string{"--offline", "--preview-hz", "241"}, []string{"TERM=dumb"}, &bytes.Buffer{}, &output, &output); err == nil {
		t.Fatal("invalid offline config succeeded")
	}
	directory := t.TempDir()
	if err := run(context.Background(), []string{
		"--config", filepath.Join(directory, "missing-user.toml"),
		"--workspace-config", filepath.Join(directory, "missing-workspace.toml"),
		"--profile", "missing",
	}, []string{"TERM=dumb"}, &bytes.Buffer{}, &output, &output); err == nil {
		t.Fatal("missing live profile succeeded")
	}
}

func TestRunDefaultsWorkspaceToCurrentDirectory(t *testing.T) {
	directory := t.TempDir()
	prompt := filepath.Join(directory, "system.md")
	if err := os.WriteFile(prompt, []byte("You are helpful."), 0o600); err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(directory, "config.toml")
	payload := `[profile.work]
provider = "anthropic"
model = "claude-sonnet-4-6"
context_window_tokens = 200000
system_prompt_file = "` + prompt + `"
permission = "read-only"
`
	if err := os.WriteFile(config, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ANTHROPIC_API_KEY", "")
	var output bytes.Buffer
	err := run(context.Background(), []string{
		"--config", config,
		"--workspace-config", filepath.Join(directory, "missing-workspace.toml"),
		"--profile", "work",
	}, []string{"TERM=dumb"}, &bytes.Buffer{}, &output, &output)
	if err == nil || !strings.Contains(err.Error(), "credential ANTHROPIC_API_KEY is not available") {
		t.Fatalf("run error = %v, want credential validation after cwd workspace resolution", err)
	}
}

func TestLastSessionPointerRoundTripAndValidation(t *testing.T) {
	t.Parallel()
	directory := filepath.Join(t.TempDir(), "sessions")
	if id, err := readLastSessionID(directory); err != nil || id != "" {
		t.Fatalf("missing pointer = %q, %v", id, err)
	}
	if err := writeLastSessionID(directory, "sess_123"); err != nil {
		t.Fatal(err)
	}
	if id, err := readLastSessionID(directory); err != nil || id != "sess_123" {
		t.Fatalf("pointer = %q, %v", id, err)
	}
	info, err := os.Stat(filepath.Join(directory, lastSessionPointer))
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("pointer mode = %v, %v", info, err)
	}
	if err := writeLastSessionID(directory, "bad\nid"); err == nil {
		t.Fatal("multiline session ID succeeded")
	}
	if err := os.WriteFile(filepath.Join(directory, lastSessionPointer), []byte(" bad "), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readLastSessionID(directory); err == nil {
		t.Fatal("malformed pointer succeeded")
	}
}
