package app

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

func TestFileEditorPreparationAndEnvironment(t *testing.T) {
	t.Parallel()
	if environmentValue([]string{"A=1", "EDITOR=/usr/bin/true"}, "A") != "1" || environmentValue(nil, "A") != "" {
		t.Fatal("environment lookup incorrect")
	}
	missing := NewFileEditor(nil)
	if _, _, err := missing.Prepare(context.Background(), "draft"); err == nil {
		t.Fatal("missing editor succeeded")
	}
	editor := NewFileEditor([]string{"VISUAL=/usr/bin/true", "EDITOR=/usr/bin/false"})
	command, read, err := editor.Prepare(context.Background(), "draft text")
	if err != nil {
		t.Fatal(err)
	}
	if command.Path != "/usr/bin/true" || len(command.Args) != 2 {
		t.Fatalf("command = %#v", command)
	}
	value, err := read()
	if err != nil || value != "draft text" {
		t.Fatalf("draft = %q, %v", value, err)
	}
	if _, err := os.Stat(command.Args[1]); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("draft was not removed: %v", err)
	}
	// The read closure remains honest if the editor removed its draft.
	if _, err := read(); err == nil {
		t.Fatal("second read unexpectedly succeeded")
	}
}

type terminalEditorFunc func(context.Context, string) (*exec.Cmd, func() (string, error), error)

func (f terminalEditorFunc) Prepare(ctx context.Context, draft string) (*exec.Cmd, func() (string, error), error) {
	return f(ctx, draft)
}

func TestTerminalEditorReducerPhases(t *testing.T) {
	t.Parallel()
	model := readyModel(t, nil)
	model.composer.SetValue("draft")
	model.terminalEditor = terminalEditorFunc(func(_ context.Context, draft string) (*exec.Cmd, func() (string, error), error) {
		if draft != "draft" {
			t.Fatalf("draft = %q", draft)
		}
		return exec.Command("/usr/bin/true"), func() (string, error) { return "edited", nil }, nil
	})
	_, prepare := model.Update(key('e', "e", tea.ModCtrl))
	prepared := prepare()
	_, executeCommand := model.Update(prepared)
	if executeCommand == nil {
		t.Fatal("terminal editor did not produce Bubble Tea exec command")
	}
	preparedValue := prepared.(editorPreparedMsg)
	_, readCommand := model.Update(editorExecutedMsg{read: preparedValue.read})
	execute(t, model, readCommand)
	if model.composer.Value() != "edited" {
		t.Fatalf("edited value = %q", model.composer.Value())
	}

	model.terminalEditor = terminalEditorFunc(func(context.Context, string) (*exec.Cmd, func() (string, error), error) {
		return nil, nil, errors.New("prepare")
	})
	_, prepare = model.Update(key('e', "e", tea.ModCtrl))
	model.Update(prepare())
	if !strings.Contains(model.banner, "prepare") {
		t.Fatalf("prepare error = %q", model.banner)
	}
	model.inflight++
	model.Update(editorExecutedMsg{err: errors.New("execute")})
	if !strings.Contains(model.banner, "execute") {
		t.Fatalf("execute error = %q", model.banner)
	}
	model.inflight++
	_, readCommand = model.Update(editorExecutedMsg{read: func() (string, error) { return "", errors.New("read") }})
	execute(t, model, readCommand)
	if !strings.Contains(model.banner, "read") {
		t.Fatalf("read error = %q", model.banner)
	}
}
