// Package config loads and resolves the standard launcher's credential-free
// TOML configuration. Later files override earlier files field by field.
package config

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pelletier/go-toml/v2"

	"github.com/regularkevvv/agentic/tui/app"
	"github.com/regularkevvv/agentic/tui/render"
)

type UI struct {
	AlternateScreen string `toml:"alternate_screen"`
	Color           string `toml:"color"`
	Thinking        string `toml:"thinking"`
	ToolDetails     string `toml:"tool_details"`
	PreviewHz       int    `toml:"preview_hz"`
}

type Session struct {
	Directory  string `toml:"directory"`
	ResumeLast *bool  `toml:"resume_last"`
}

type Workspace struct {
	Root string `toml:"root"`
}

type Profile struct {
	Provider            string    `toml:"provider"`
	Model               string    `toml:"model"`
	ContextWindowTokens int       `toml:"context_window_tokens"`
	SystemPromptFile    string    `toml:"system_prompt_file"`
	Permission          string    `toml:"permission"`
	Workspace           Workspace `toml:"workspace"`
}

type Document struct {
	UI       UI                 `toml:"ui"`
	Session  Session            `toml:"session"`
	Profiles map[string]Profile `toml:"profile"`
}

type Flags struct {
	Provider            string
	Model               string
	ContextWindowTokens int
	SystemPromptFile    string
	Permission          string
	WorkspaceRoot       string
	SessionDirectory    string
	AlternateScreen     string
	Color               string
	Thinking            string
	ToolDetails         string
	PreviewHz           int
}

type Resolved struct {
	ProfileName         string
	Provider            string
	Model               string
	ContextWindowTokens int
	SystemPromptFile    string
	Permission          string
	WorkspaceRoot       string
	SessionDirectory    string
	ResumeLast          bool
	UI                  app.Config
}

func Load(paths ...string) (Document, error) {
	result := Document{Profiles: make(map[string]Profile)}
	for _, path := range paths {
		if path == "" {
			continue
		}
		payload, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return Document{}, fmt.Errorf("read config %s: %w", path, err)
		}
		var next Document
		if err := toml.Unmarshal(payload, &next); err != nil {
			return Document{}, fmt.Errorf("decode config %s: %w", path, err)
		}
		merge(&result, next)
	}
	return result, nil
}

func Resolve(document Document, profileName string, flags Flags, environ []string) (Resolved, error) {
	if profileName == "" {
		return Resolved{}, errors.New("profile name is required")
	}
	profile, found := document.Profiles[profileName]
	if !found {
		return Resolved{}, fmt.Errorf("profile %q was not found", profileName)
	}
	resolved := Resolved{
		ProfileName: profileName, Provider: profile.Provider, Model: profile.Model,
		ContextWindowTokens: profile.ContextWindowTokens, SystemPromptFile: profile.SystemPromptFile,
		Permission: profile.Permission, WorkspaceRoot: profile.Workspace.Root,
		SessionDirectory: document.Session.Directory, ResumeLast: true,
	}
	if document.Session.ResumeLast != nil {
		resolved.ResumeLast = *document.Session.ResumeLast
	}
	applyFlags(&resolved, flags)
	if resolved.Provider == "" || resolved.Model == "" || resolved.ContextWindowTokens <= 0 {
		return Resolved{}, errors.New("profile provider, model, and positive context_window_tokens are required")
	}
	if resolved.SystemPromptFile == "" {
		return Resolved{}, errors.New("profile system_prompt_file is required")
	}
	if resolved.WorkspaceRoot == "" {
		return Resolved{}, errors.New("profile workspace root is required")
	}
	if resolved.Permission == "" {
		resolved.Permission = "workspace-write"
	}
	if resolved.Permission != "read-only" && resolved.Permission != "workspace-write" && resolved.Permission != "custom" {
		return Resolved{}, fmt.Errorf("unsupported permission policy %q", resolved.Permission)
	}
	var err error
	resolved.WorkspaceRoot, err = absoluteExpanded(resolved.WorkspaceRoot)
	if err != nil {
		return Resolved{}, fmt.Errorf("workspace root: %w", err)
	}
	resolved.SystemPromptFile, err = absoluteExpanded(resolved.SystemPromptFile)
	if err != nil {
		return Resolved{}, fmt.Errorf("system prompt file: %w", err)
	}
	if resolved.SessionDirectory == "" {
		resolved.SessionDirectory, err = defaultSessionDirectory(resolved.WorkspaceRoot)
	} else {
		resolved.SessionDirectory, err = absoluteExpanded(resolved.SessionDirectory)
	}
	if err != nil {
		return Resolved{}, fmt.Errorf("session directory: %w", err)
	}
	resolved.UI, err = resolveUI(document.UI, flags, environ)
	if err != nil {
		return Resolved{}, err
	}
	return resolved, nil
}

func UserPath() (string, error) {
	directory, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(directory, "agentic", "config.toml"), nil
}

func WorkspacePath(workspace string) string { return filepath.Join(workspace, ".agentic.toml") }

func merge(target *Document, source Document) {
	mergeUI(&target.UI, source.UI)
	if source.Session.Directory != "" {
		target.Session.Directory = source.Session.Directory
	}
	if source.Session.ResumeLast != nil {
		value := *source.Session.ResumeLast
		target.Session.ResumeLast = &value
	}
	if target.Profiles == nil {
		target.Profiles = make(map[string]Profile)
	}
	for name, profile := range source.Profiles {
		current := target.Profiles[name]
		mergeProfile(&current, profile)
		target.Profiles[name] = current
	}
}

func mergeUI(target *UI, source UI) {
	if source.AlternateScreen != "" {
		target.AlternateScreen = source.AlternateScreen
	}
	if source.Color != "" {
		target.Color = source.Color
	}
	if source.Thinking != "" {
		target.Thinking = source.Thinking
	}
	if source.ToolDetails != "" {
		target.ToolDetails = source.ToolDetails
	}
	if source.PreviewHz != 0 {
		target.PreviewHz = source.PreviewHz
	}
}

func mergeProfile(target *Profile, source Profile) {
	if source.Provider != "" {
		target.Provider = source.Provider
	}
	if source.Model != "" {
		target.Model = source.Model
	}
	if source.ContextWindowTokens != 0 {
		target.ContextWindowTokens = source.ContextWindowTokens
	}
	if source.SystemPromptFile != "" {
		target.SystemPromptFile = source.SystemPromptFile
	}
	if source.Permission != "" {
		target.Permission = source.Permission
	}
	if source.Workspace.Root != "" {
		target.Workspace.Root = source.Workspace.Root
	}
}

func applyFlags(target *Resolved, flags Flags) {
	if flags.Provider != "" {
		target.Provider = flags.Provider
	}
	if flags.Model != "" {
		target.Model = flags.Model
	}
	if flags.ContextWindowTokens != 0 {
		target.ContextWindowTokens = flags.ContextWindowTokens
	}
	if flags.SystemPromptFile != "" {
		target.SystemPromptFile = flags.SystemPromptFile
	}
	if flags.Permission != "" {
		target.Permission = flags.Permission
	}
	if flags.WorkspaceRoot != "" {
		target.WorkspaceRoot = flags.WorkspaceRoot
	}
	if flags.SessionDirectory != "" {
		target.SessionDirectory = flags.SessionDirectory
	}
}

func resolveUI(file UI, flags Flags, environ []string) (app.Config, error) {
	config := app.DefaultConfig()
	alt := first(flags.AlternateScreen, file.AlternateScreen, string(app.AlternateAuto))
	color := first(flags.Color, file.Color, "auto")
	thinking := first(flags.Thinking, file.Thinking, string(render.ThinkingCollapsed))
	tools := first(flags.ToolDetails, file.ToolDetails, "collapsed")
	preview := flags.PreviewHz
	if preview == 0 {
		preview = file.PreviewHz
	}
	if preview == 0 {
		preview = 60
	}
	config.AlternateScreen = app.AlternateScreen(alt)
	config.Thinking = render.ThinkingMode(thinking)
	config.ToolDetails = tools == "expanded"
	config.PreviewHz = preview
	switch color {
	case "always":
		config.NoColor = false
	case "never":
		config.NoColor = true
	case "auto":
		config.NoColor = hasEnvironment(environ, "NO_COLOR")
	default:
		return app.Config{}, fmt.Errorf("invalid color mode %q", color)
	}
	if tools != "collapsed" && tools != "expanded" {
		return app.Config{}, fmt.Errorf("invalid tool_details mode %q", tools)
	}
	if err := config.Validate(); err != nil {
		return app.Config{}, err
	}
	return config, nil
}

func defaultSessionDirectory(workspace string) (string, error) {
	directory, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256([]byte(filepath.Clean(workspace)))
	return filepath.Join(directory, "agentic", "sessions", fmt.Sprintf("%x", digest[:8])), nil
}

func absoluteExpanded(value string) (string, error) {
	if strings.HasPrefix(value, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		value = filepath.Join(home, strings.TrimPrefix(value, "~/"))
	}
	return filepath.Abs(value)
}

func first(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func hasEnvironment(environ []string, key string) bool {
	for _, value := range environ {
		if value == key || strings.HasPrefix(value, key+"=") {
			return true
		}
	}
	return false
}
