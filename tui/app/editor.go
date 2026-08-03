package app

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
)

type fileEditor struct{ environ []string }

// NewFileEditor prepares $VISUAL or $EDITOR in a temporary draft file. The
// actual process is returned through Bubble Tea's terminal-releasing Exec port.
func NewFileEditor(environ []string) TerminalEditor {
	return &fileEditor{environ: append([]string(nil), environ...)}
}

func (e *fileEditor) Prepare(_ context.Context, draft string) (*exec.Cmd, func() (string, error), error) {
	editor := environmentValue(e.environ, "VISUAL")
	if editor == "" {
		editor = environmentValue(e.environ, "EDITOR")
	}
	fields := strings.Fields(editor)
	if len(fields) == 0 {
		return nil, nil, errors.New("VISUAL or EDITOR is required for the external editor")
	}
	file, err := os.CreateTemp("", "agentic-draft-*.md")
	if err != nil {
		return nil, nil, err
	}
	path := file.Name()
	cleanup := func() { _ = os.Remove(path) }
	if _, err := file.WriteString(draft); err != nil {
		_ = file.Close()
		cleanup()
		return nil, nil, err
	}
	if err := file.Close(); err != nil {
		cleanup()
		return nil, nil, err
	}
	command := exec.Command(fields[0], append(fields[1:], path)...)
	read := func() (string, error) {
		defer cleanup()
		payload, err := os.ReadFile(path)
		return string(payload), err
	}
	return command, read, nil
}

func environmentValue(environ []string, key string) string {
	for _, value := range environ {
		name, item, found := strings.Cut(value, "=")
		if found && name == key {
			return item
		}
	}
	return ""
}
