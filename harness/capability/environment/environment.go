// Package environment provides ordinary capability tools over the injected
// session Environment. Governance remains a separate permission capability.
package environment

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

const (
	ID = "environment"

	ToolReadFile      = "read_file"
	ToolWriteFile     = "write_file"
	ToolListFiles     = "list_files"
	ToolStatFile      = "stat_file"
	ToolMakeDirectory = "make_directory"
	ToolRemovePath    = "remove_path"
	ToolRunCommand    = "run_command"
)

type Config struct {
	Files bool
	Shell bool
}

type Capability struct {
	config Config
}

func New(config Config) (*Capability, error) {
	if !config.Files && !config.Shell {
		return nil, errors.New("environment capability must enable files or shell")
	}
	return &Capability{config: config}, nil
}

func (c *Capability) ID() string                    { return ID }
func (c *Capability) Ordering() capability.Ordering { return capability.Ordering{} }

func (c *Capability) Register(registry *capability.Registry) error {
	toolset := agentic.NewToolset()
	if c.config.Files {
		if err := addFileTools(toolset); err != nil {
			return err
		}
	}
	if c.config.Shell {
		if err := addShellTool(toolset); err != nil {
			return err
		}
	}
	if err := registry.AddToolset(toolset); err != nil {
		return err
	}
	if c.config.Files {
		for _, definition := range []struct {
			name   string
			action string
		}{
			{ToolReadFile, "read"},
			{ToolWriteFile, "write"},
			{ToolListFiles, "list"},
			{ToolStatFile, "stat"},
			{ToolMakeDirectory, "mkdir"},
			{ToolRemovePath, "remove"},
		} {
			if err := registry.AddEffectResolver(definition.name, fileEffect(definition.action)); err != nil {
				return err
			}
		}
	}
	if c.config.Shell {
		if err := registry.AddEffectResolver(ToolRunCommand, capability.EffectResolverFunc(commandEffect)); err != nil {
			return err
		}
	}
	return nil
}

type pathInput struct {
	Path string `json:"path" description:"Path relative to the session workspace"`
}

type writeFileInput struct {
	Path    string `json:"path" description:"Path relative to the session workspace"`
	Content string `json:"content" description:"UTF-8 file content"`
	Mode    uint32 `json:"mode,omitempty" description:"Optional Unix permission bits; defaults to 0644"`
}

type directoryInput struct {
	Path string `json:"path" description:"Directory path relative to the session workspace"`
	Mode uint32 `json:"mode,omitempty" description:"Optional Unix permission bits; defaults to 0755"`
}

type commandInput struct {
	Name  string   `json:"name" description:"Executable name"`
	Args  []string `json:"args,omitempty" description:"Argument vector without argv[0]"`
	Dir   string   `json:"dir,omitempty" description:"Working directory relative to the environment cwd"`
	Env   []string `json:"env,omitempty" description:"Additional KEY=VALUE environment entries"`
	Stdin string   `json:"stdin,omitempty" description:"UTF-8 standard input"`
}

type readFileOutput struct {
	Content string `json:"content"`
	Bytes   int    `json:"bytes"`
}

type writeOutput struct {
	Path  string `json:"path"`
	Bytes int    `json:"bytes"`
}

type commandOutput struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exit_code"`
}

func addFileTools(toolset *agentic.FuncToolset) error {
	readTool, readHandler, err := agentic.ToolWithContext(
		ToolReadFile,
		"Read a UTF-8 file from the session workspace.",
		func(ctx context.Context, input pathInput) (readFileOutput, error) {
			runtime, err := requireRuntime(ctx)
			if err != nil {
				return readFileOutput{}, err
			}
			data, err := runtime.Environment.Files().ReadFile(ctx, input.Path)
			if err != nil {
				return readFileOutput{}, err
			}
			return readFileOutput{Content: strings.ToValidUTF8(string(data), "�"), Bytes: len(data)}, nil
		},
	)
	if err != nil {
		return err
	}
	toolset.Add(readTool, readHandler)

	writeTool, writeHandler, err := agentic.ToolWithContext(
		ToolWriteFile,
		"Write UTF-8 content to a file in the session workspace.",
		func(ctx context.Context, input writeFileInput) (writeOutput, error) {
			runtime, err := requireRuntime(ctx)
			if err != nil {
				return writeOutput{}, err
			}
			mode, err := fileMode(input.Mode, 0o644)
			if err != nil {
				return writeOutput{}, err
			}
			data := []byte(input.Content)
			if err := runtime.Environment.Files().WriteFile(ctx, input.Path, data, mode); err != nil {
				return writeOutput{}, err
			}
			return writeOutput{Path: input.Path, Bytes: len(data)}, nil
		},
	)
	if err != nil {
		return err
	}
	toolset.Add(writeTool, writeHandler)

	listTool, listHandler, err := agentic.ToolWithContext(
		ToolListFiles,
		"List one directory in the session workspace.",
		func(ctx context.Context, input pathInput) ([]env.DirEntry, error) {
			runtime, err := requireRuntime(ctx)
			if err != nil {
				return nil, err
			}
			return runtime.Environment.Files().ReadDir(ctx, input.Path)
		},
	)
	if err != nil {
		return err
	}
	toolset.Add(listTool, listHandler)

	statTool, statHandler, err := agentic.ToolWithContext(
		ToolStatFile,
		"Inspect one file or directory in the session workspace.",
		func(ctx context.Context, input pathInput) (env.FileInfo, error) {
			runtime, err := requireRuntime(ctx)
			if err != nil {
				return env.FileInfo{}, err
			}
			return runtime.Environment.Files().Stat(ctx, input.Path)
		},
	)
	if err != nil {
		return err
	}
	toolset.Add(statTool, statHandler)

	mkdirTool, mkdirHandler, err := agentic.ToolWithContext(
		ToolMakeDirectory,
		"Create a directory and missing parents in the session workspace.",
		func(ctx context.Context, input directoryInput) (string, error) {
			runtime, err := requireRuntime(ctx)
			if err != nil {
				return "", err
			}
			mode, err := fileMode(input.Mode, 0o755)
			if err != nil {
				return "", err
			}
			if err := runtime.Environment.Files().MkdirAll(ctx, input.Path, mode); err != nil {
				return "", err
			}
			return input.Path, nil
		},
	)
	if err != nil {
		return err
	}
	toolset.Add(mkdirTool, mkdirHandler)

	removeTool, removeHandler, err := agentic.ToolWithContext(
		ToolRemovePath,
		"Remove one file or empty directory from the session workspace.",
		func(ctx context.Context, input pathInput) (string, error) {
			runtime, err := requireRuntime(ctx)
			if err != nil {
				return "", err
			}
			if err := runtime.Environment.Files().Remove(ctx, input.Path); err != nil {
				return "", err
			}
			return input.Path, nil
		},
	)
	if err != nil {
		return err
	}
	toolset.Add(removeTool, removeHandler)
	return nil
}

func addShellTool(toolset *agentic.FuncToolset) error {
	tool, handler, err := agentic.ToolWithContext(
		ToolRunCommand,
		"Run one command through the session environment. Local environments are not OS sandboxes.",
		func(ctx context.Context, input commandInput) (commandOutput, error) {
			runtime, err := requireRuntime(ctx)
			if err != nil {
				return commandOutput{}, err
			}
			if input.Name == "" {
				return commandOutput{}, errors.New("command name is required")
			}
			shell, ok := runtime.Environment.Shell()
			if !ok {
				return commandOutput{}, &env.Error{Code: env.CodeUnsupported, Op: "exec", Err: errors.New("environment has no shell")}
			}
			result, err := shell.Exec(ctx, env.Command{
				Name:  input.Name,
				Args:  append([]string(nil), input.Args...),
				Dir:   input.Dir,
				Env:   append([]string(nil), input.Env...),
				Stdin: []byte(input.Stdin),
			})
			output := commandOutput{
				Stdout:   strings.ToValidUTF8(string(result.Stdout), "�"),
				Stderr:   strings.ToValidUTF8(string(result.Stderr), "�"),
				ExitCode: result.ExitCode,
			}
			return output, err
		},
	)
	if err != nil {
		return err
	}
	toolset.Add(tool, handler)
	return nil
}

func requireRuntime(ctx context.Context) (harnessruntime.ToolRuntime, error) {
	runtime, ok := harnessruntime.FromContext(ctx)
	if !ok || runtime.Environment == nil {
		return harnessruntime.ToolRuntime{}, errors.New("environment tool requires harness ToolRuntime")
	}
	return runtime, nil
}

func fileMode(value uint32, fallback fs.FileMode) (fs.FileMode, error) {
	if value == 0 {
		return fallback, nil
	}
	if value&^uint32(0o777) != 0 {
		return 0, errors.New("file mode may contain only permission bits")
	}
	return fs.FileMode(value), nil
}

func fileEffect(action string) capability.EffectResolver {
	return capability.EffectResolverFunc(func(
		ctx context.Context,
		call agentic.ToolUse,
		environment env.Environment,
	) (capability.Effect, error) {
		pathValue, ok := call.Input["path"].(string)
		if !ok || pathValue == "" {
			return capability.Effect{}, errors.New("path input is required")
		}
		resource, err := environment.Files().CanonicalPath(ctx, pathValue)
		return capability.Effect{
			Capability: "filesystem",
			Action:     action,
			Resource:   resource,
		}, err
	})
}

func commandEffect(
	_ context.Context,
	call agentic.ToolUse,
	_ env.Environment,
) (capability.Effect, error) {
	name, ok := call.Input["name"].(string)
	if !ok || name == "" {
		return capability.Effect{}, errors.New("command name is required")
	}
	payload, err := json.Marshal(struct {
		Name string
		Args any
		Dir  any
	}{Name: name, Args: call.Input["args"], Dir: call.Input["dir"]})
	if err != nil {
		return capability.Effect{}, fmt.Errorf("canonicalize command: %w", err)
	}
	display := commandSummary(call.Input)
	if display == "" {
		display = name
	}
	return capability.Effect{
		Capability: "shell",
		Action:     "exec",
		Resource: env.CanonicalResource{
			Scheme:  "command",
			ID:      string(payload),
			Display: display,
		},
	}, nil
}

var (
	embeddedSecret = regexp.MustCompile(`(?i)(api[-_]?key|access[-_]?token|auth(?:orization)?|cookie|credential|password|passwd|private[-_]?key|client[-_]?secret)=([^&\s]+)`)
	bearerSecret   = regexp.MustCompile(`(?i)\bbearer\s+[^\s]+`)
)

// ToolSummary is the environment capability's safe presentation policy. It
// intentionally omits command environment variables, stdin, and file content.
// Applications assembling different tools can provide their own summarizer in
// harness.RuntimeConfig.
func ToolSummary(call agentic.ToolUse) string {
	switch call.Name {
	case ToolReadFile, ToolWriteFile, ToolListFiles, ToolStatFile, ToolMakeDirectory, ToolRemovePath:
		path, _ := call.Input["path"].(string)
		return path
	case ToolRunCommand:
		return commandSummary(call.Input)
	default:
		return ""
	}
}

func commandSummary(input map[string]any) string {
	name, _ := input["name"].(string)
	if name == "" {
		return ""
	}
	words := []string{name}
	args := stringSlice(input["args"])
	redactNext, redactHeader := false, false
	for _, arg := range args {
		switch {
		case redactNext:
			words = append(words, "[redacted]")
			redactNext = false
			continue
		case redactHeader:
			words = append(words, safeHeader(arg))
			redactHeader = false
			continue
		}
		flag, value, assigned := strings.Cut(arg, "=")
		if assigned && isHeaderFlag(flag) {
			words = append(words, flag+"="+safeHeader(value))
			continue
		}
		if assigned && isSecretLabel(flag) {
			words = append(words, flag+"=[redacted]")
			continue
		}
		if isHeaderFlag(arg) {
			words = append(words, arg)
			redactHeader = true
			continue
		}
		if isSecretFlag(arg) {
			words = append(words, arg)
			redactNext = true
			continue
		}
		words = append(words, redactEmbedded(arg))
	}
	for index := range words {
		words[index] = displayWord(words[index])
	}
	result := strings.Join(words, " ")
	if dir, _ := input["dir"].(string); dir != "" {
		result += " · in " + displayWord(dir)
	}
	return result
}

func stringSlice(value any) []string {
	switch values := value.(type) {
	case []string:
		return append([]string(nil), values...)
	case []any:
		result := make([]string, 0, len(values))
		for _, value := range values {
			if text, ok := value.(string); ok {
				result = append(result, text)
			}
		}
		return result
	default:
		return nil
	}
}

func isHeaderFlag(value string) bool { return value == "-H" || value == "--header" }

func isSecretFlag(value string) bool {
	if value == "-u" || value == "--user" {
		return true
	}
	return strings.HasPrefix(value, "-") && isSecretLabel(value)
}

func isSecretLabel(value string) bool {
	var normalized strings.Builder
	for _, current := range strings.ToLower(strings.TrimLeft(value, "-")) {
		if unicode.IsLetter(current) || unicode.IsDigit(current) {
			normalized.WriteRune(current)
		}
	}
	label := normalized.String()
	for _, fragment := range []string{
		"apikey", "accesstoken", "authorization", "auth", "cookie", "credential",
		"password", "passwd", "privatekey", "clientsecret", "secret", "token",
	} {
		if strings.Contains(label, fragment) {
			return true
		}
	}
	return false
}

func safeHeader(value string) string {
	name, _, found := strings.Cut(value, ":")
	if found && isSecretLabel(name) {
		return name + ": [redacted]"
	}
	return redactEmbedded(value)
}

func redactEmbedded(value string) string {
	value = embeddedSecret.ReplaceAllString(value, "$1=[redacted]")
	return bearerSecret.ReplaceAllString(value, "Bearer [redacted]")
}

func displayWord(value string) string {
	if value == "" {
		return `''`
	}
	for _, current := range value {
		if !unicode.IsLetter(current) && !unicode.IsDigit(current) &&
			!strings.ContainsRune("_@%+=:,./-[]", current) {
			return strconv.Quote(value)
		}
	}
	return value
}

var _ capability.Capability = (*Capability)(nil)
