package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	tea "charm.land/bubbletea/v2"

	"github.com/regularkevvv/agentic/tui/app"
	appconfig "github.com/regularkevvv/agentic/tui/config"
	"github.com/regularkevvv/agentic/tui/internal/testhost"
	"github.com/regularkevvv/agentic/tui/standard"
)

type flags struct {
	offline, noAltScreen, noColor                                    bool
	profile, configPath, workspaceConfig, resumeID                   string
	provider, model, systemPrompt, permission, workspace, sessionDir string
	contextWindow, previewHz                                         int
	thinking, toolDetails, color, alternateScreen                    string
}

func parseFlags(arguments []string, output io.Writer) (flags, error) {
	var values flags
	set := flag.NewFlagSet("agentic-harness", flag.ContinueOnError)
	set.SetOutput(output)
	set.BoolVar(&values.offline, "offline", false, "run the credential-free deterministic host")
	set.BoolVar(&values.noAltScreen, "no-alt-screen", false, "preserve terminal scrollback")
	set.BoolVar(&values.noColor, "no-color", false, "disable ANSI color")
	set.StringVar(&values.profile, "profile", "work", "named model profile")
	set.StringVar(&values.configPath, "config", "", "user config path")
	set.StringVar(&values.workspaceConfig, "workspace-config", "", "workspace config path")
	set.StringVar(&values.resumeID, "resume", "", "resume a durable session ID")
	set.StringVar(&values.provider, "provider", "", "override provider ID")
	set.StringVar(&values.model, "model", "", "override model ID")
	set.IntVar(&values.contextWindow, "context-window", 0, "override context window tokens")
	set.StringVar(&values.systemPrompt, "system-prompt-file", "", "override system prompt file")
	set.StringVar(&values.permission, "permission", "", "read-only or workspace-write")
	set.StringVar(&values.workspace, "workspace", "", "override workspace root (defaults to current directory)")
	set.StringVar(&values.sessionDir, "session-dir", "", "override durable session directory")
	set.IntVar(&values.previewHz, "preview-hz", 0, "preview coalescing rate")
	set.StringVar(&values.thinking, "thinking", "", "visible, collapsed, or hidden")
	set.StringVar(&values.toolDetails, "tool-details", "", "collapsed or expanded")
	set.StringVar(&values.color, "color", "", "auto, always, or never")
	set.StringVar(&values.alternateScreen, "alternate-screen", "", "auto, always, or never")
	if err := set.Parse(arguments); err != nil {
		return flags{}, err
	}
	if set.NArg() != 0 {
		return flags{}, fmt.Errorf("unexpected arguments: %v", set.Args())
	}
	return values, nil
}

func run(ctx context.Context, arguments, environ []string, input io.Reader, output, errorsOutput io.Writer) error {
	values, err := parseFlags(arguments, errorsOutput)
	if err != nil {
		return err
	}
	if values.offline {
		config := app.DefaultConfig()
		if values.noAltScreen {
			config.AlternateScreen = app.AlternateNever
		}
		if values.noColor {
			config.NoColor = true
		}
		if values.previewHz != 0 {
			config.PreviewHz = values.previewHz
		}
		_, err := app.Run(ctx, testhost.New(nil), app.Options{
			Config: config, ResumeID: values.resumeID, Environ: environ,
			TerminalEditor: app.NewFileEditor(environ),
		}, tea.WithInput(input), tea.WithOutput(output), tea.WithEnvironment(environ))
		return err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	userPath := values.configPath
	if userPath == "" {
		userPath, err = appconfig.UserPath()
		if err != nil {
			return err
		}
	}
	workspaceForConfig := cwd
	if values.workspace != "" {
		workspaceForConfig, err = filepath.Abs(values.workspace)
		if err != nil {
			return err
		}
	}
	workspacePath := values.workspaceConfig
	if workspacePath == "" {
		workspacePath = appconfig.WorkspacePath(workspaceForConfig)
	}
	document, err := appconfig.Load(userPath, workspacePath)
	if err != nil {
		return err
	}
	color := values.color
	if values.noColor {
		color = "never"
	}
	altScreen := values.alternateScreen
	if values.noAltScreen {
		altScreen = "never"
	}
	resolved, err := appconfig.Resolve(document, values.profile, appconfig.Flags{
		Provider: values.provider, Model: values.model, ContextWindowTokens: values.contextWindow,
		SystemPromptFile: values.systemPrompt, Permission: values.permission,
		WorkspaceRoot: values.workspace, WorkingDirectory: cwd, SessionDirectory: values.sessionDir,
		AlternateScreen: altScreen, Color: color, Thinking: values.thinking,
		ToolDetails: values.toolDetails, PreviewHz: values.previewHz,
	}, environ)
	if err != nil {
		return err
	}
	registry, err := standard.NewRegistry(standard.BuiltinFactories(nil)...)
	if err != nil {
		return err
	}
	assembly, err := standard.Build(ctx, registry, standard.FromResolved(resolved))
	if err != nil {
		return err
	}
	resumeID := values.resumeID
	if resumeID == "" && resolved.ResumeLast {
		resumeID, err = readLastSessionID(resolved.SessionDirectory)
		if err != nil {
			return err
		}
	}
	final, err := app.Run(ctx, assembly.Host, app.Options{
		Config: resolved.UI, ResumeID: resumeID, Environ: environ,
		TerminalEditor: app.NewFileEditor(environ),
	}, tea.WithInput(input), tea.WithOutput(output), tea.WithEnvironment(environ))
	if err != nil {
		return err
	}
	if resolved.ResumeLast && final != nil && final.SessionID() != "" {
		return writeLastSessionID(resolved.SessionDirectory, final.SessionID())
	}
	return nil
}

const lastSessionPointer = "last-session"

func readLastSessionID(directory string) (string, error) {
	payload, err := os.ReadFile(filepath.Join(directory, lastSessionPointer))
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("read last session pointer: %w", err)
	}
	id := string(payload)
	if id == "" || strings.TrimSpace(id) != id || strings.ContainsAny(id, "\r\n") {
		return "", errors.New("last session pointer is malformed")
	}
	return id, nil
}

func writeLastSessionID(directory, id string) error {
	if directory == "" || id == "" || strings.TrimSpace(id) != id || strings.ContainsAny(id, "\r\n") {
		return errors.New("session directory and a single-line session ID are required")
	}
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return fmt.Errorf("create session directory for resume pointer: %w", err)
	}
	temporary, err := os.CreateTemp(directory, ".last-session-*")
	if err != nil {
		return fmt.Errorf("create last session pointer: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("secure last session pointer: %w", err)
	}
	if _, err := temporary.WriteString(id); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write last session pointer: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync last session pointer: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close last session pointer: %w", err)
	}
	if err := os.Rename(temporaryPath, filepath.Join(directory, lastSessionPointer)); err != nil {
		return fmt.Errorf("publish last session pointer: %w", err)
	}
	return nil
}

func main() {
	if err := run(context.Background(), os.Args[1:], os.Environ(), os.Stdin, os.Stdout, os.Stderr); err != nil && !errors.Is(err, flag.ErrHelp) {
		fmt.Fprintln(os.Stderr, "agentic-harness:", err)
		os.Exit(1)
	}
}
