package sandbox

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

type execution struct {
	root    string
	cwd     string
	network bool
	command harnessenv.Command
}

func run(ctx context.Context, name string, args, environment []string, cwd string, stdin []byte) (harnessenv.CommandResult, error) {
	command := exec.CommandContext(ctx, name, args...)
	command.Dir = cwd
	command.Env = environment
	command.Stdin = bytes.NewReader(stdin)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	err := command.Run()
	result := harnessenv.CommandResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes()}
	if err == nil {
		return result, nil
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		result.ExitCode = exit.ExitCode()
		return result, nil
	}
	return harnessenv.CommandResult{}, err
}

func resolveCommand(name, cwd string) (string, error) {
	candidate := name
	if !filepath.IsAbs(candidate) && strings.ContainsRune(candidate, filepath.Separator) {
		candidate = filepath.Join(cwd, candidate)
	}
	target, err := exec.LookPath(candidate)
	if err != nil {
		return "", err
	}
	if !filepath.IsAbs(target) {
		target = filepath.Join(cwd, target)
	}
	if resolved, resolveErr := filepath.EvalSymlinks(target); resolveErr == nil {
		target = resolved
	}
	return target, nil
}

func commandHelpers(target string) string {
	if filepath.Base(target) == "go" && filepath.Base(filepath.Dir(target)) == "bin" {
		return filepath.Dir(filepath.Dir(target))
	}
	if target == "/usr/bin/git" {
		return "/usr/libexec/git-core"
	}
	return filepath.Dir(target)
}

func commandEnvironment(additions []string, forced map[string]string) []string {
	values := make(map[string]string)
	add := func(entry string) {
		name, value, ok := strings.Cut(entry, "=")
		if ok && name != "" {
			values[name] = value
		}
	}
	for _, entry := range os.Environ() {
		add(entry)
	}
	for _, entry := range additions {
		add(entry)
	}
	for name, value := range forced {
		values[name] = value
	}
	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make([]string, 0, len(names))
	for _, name := range names {
		result = append(result, name+"="+values[name])
	}
	return result
}

func privateTemp() (string, error) {
	path, err := os.MkdirTemp("", "agentic-sandbox-")
	if err != nil {
		return "", err
	}
	return canonicalPrivateTemp(path)
}

func canonicalPrivateTemp(path string) (string, error) {
	canonical, err := filepath.EvalSymlinks(path)
	if err != nil {
		_ = os.RemoveAll(path)
		return "", err
	}
	return filepath.Clean(canonical), nil
}
