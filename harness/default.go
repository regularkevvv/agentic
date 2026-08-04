package harness

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	agentic "github.com/regularkevvv/agentic"

	artifactfile "github.com/regularkevvv/agentic/harness/artifact/file"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	artifactcapability "github.com/regularkevvv/agentic/harness/capability/artifacts"
	environmentcapability "github.com/regularkevvv/agentic/harness/capability/environment"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
	envlocal "github.com/regularkevvv/agentic/harness/env/local"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/permission"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	storejsonl "github.com/regularkevvv/agentic/harness/store/jsonl"
)

// DefaultConfig names every host location and model geometry assumption used
// by the convenient local assembly.
type DefaultConfig struct {
	WorkspaceRoot       string
	SessionDir          string
	ContextWindowTokens int

	ToolCancellationGrace time.Duration
	TokenCounter          contextpolicy.TokenCounter
	RecentMessages        int
	AdditionalToolSchemas []agentic.Tool

	SpillThreshold       int
	SpillHead            int
	SpillTail            int
	DisableArtifactSpill bool
	PromptCacheRetention agentic.PromptCacheRetention
	ModelStreaming       bool
	PermissionPolicy     *permission.Policy
}

// DefaultAssembly is the concrete substrate plus ordinary exported
// capabilities used by Default. Applications may inspect and extend this
// assembly before passing it to New.
type DefaultAssembly struct {
	Runtime      RuntimeConfig
	Capabilities []Capability
}

// AssembleDefault constructs the replaceable local adapters and ordinary
// public capabilities behind Default.
func AssembleDefault(config DefaultConfig) (DefaultAssembly, error) {
	workspace, sessions, err := validateDefaultPaths(config)
	if err != nil {
		return DefaultAssembly{}, err
	}
	grace := config.ToolCancellationGrace
	if grace == 0 {
		grace = time.Second
	}
	if grace < 0 {
		return DefaultAssembly{}, errors.New("tool cancellation grace cannot be negative")
	}
	spillConfig := spill.Config{
		Threshold: config.SpillThreshold,
		Head:      config.SpillHead,
		Tail:      config.SpillTail,
		Disabled:  config.DisableArtifactSpill,
	}
	if err := spillConfig.Validate(); err != nil {
		return DefaultAssembly{}, err
	}
	journals, err := storejsonl.New(filepath.Join(sessions, "journals"))
	if err != nil {
		return DefaultAssembly{}, err
	}
	artifacts, err := artifactfile.New(filepath.Join(sessions, "artifacts"))
	if err != nil {
		return DefaultAssembly{}, err
	}
	environments, err := envlocal.NewFactory(envlocal.Config{
		Root:     workspace,
		Cwd:      ".",
		Symlinks: envlocal.SymlinkWithinRoot,
	})
	if err != nil {
		return DefaultAssembly{}, err
	}
	processors, err := spill.NewFactory(artifacts, spillConfig)
	if err != nil {
		return DefaultAssembly{}, err
	}
	environmentCapability, err := environmentcapability.New(environmentcapability.Config{Files: true, Shell: true})
	if err != nil {
		return DefaultAssembly{}, err
	}
	readLimit := artifactReadLimit(config)
	artifactCapability, err := artifactcapability.New(artifactcapability.Config{
		Store:        artifacts,
		MaxReadBytes: readLimit,
	})
	if err != nil {
		return DefaultAssembly{}, err
	}
	structured, err := contextpolicy.NewStructuredCompactor(contextpolicy.StructuredConfig{
		MaxSummaryBytes: defaultSummaryBytes(config.ContextWindowTokens),
		MaxEntryBytes:   min(512, defaultSummaryBytes(config.ContextWindowTokens)),
	})
	if err != nil {
		return DefaultAssembly{}, err
	}
	contextCapability := capability.Func{
		Name: "contextpolicy",
		Apply: func(registry *capability.Registry) error {
			return registry.ConfigureContext(contextpolicy.Config{
				ContextWindowTokens: config.ContextWindowTokens,
				TriggerPercent:      70,
				TargetPercent:       50,
				RecentMessages:      config.RecentMessages,
				Counter:             config.TokenCounter,
				Tools:               append([]agentic.Tool(nil), config.AdditionalToolSchemas...),
			}, structured)
		},
	}
	permissionPolicy := config.PermissionPolicy
	if permissionPolicy == nil {
		permissionPolicy = permission.WorkspaceWrite()
	}
	permissionCapability, err := permission.NewCapability(
		permissionPolicy,
		permission.WithOrdering(capability.Ordering{
			After: []string{environmentcapability.ID, artifactcapability.ID},
		}),
	)
	if err != nil {
		return DefaultAssembly{}, err
	}
	return DefaultAssembly{
		Runtime: RuntimeConfig{
			Sessions:              journals,
			Codec:                 jsoncodec.New(),
			Events:                inproc.NewFactory(),
			Environments:          environments,
			ResultProcessors:      processors,
			Clock:                 system.NewClock(),
			IDs:                   system.NewIDs(),
			ToolCancellationGrace: grace,
			PromptCacheRetention:  config.PromptCacheRetention,
			ModelStreaming:        config.ModelStreaming,
			ToolSummarizer:        environmentcapability.ToolSummary,
		},
		Capabilities: []Capability{
			environmentCapability,
			artifactCapability,
			contextCapability,
			permissionCapability,
		},
	}, nil
}

// Default constructs the experimental v0.1 local harness. Local shell commands
// run as the host user; this assembly is governance policy, not an OS sandbox.
func Default[O any](runner agentic.Runner[O], config DefaultConfig) (*Harness[O], error) {
	if _, err := agentic.RequireDriver(runner); err != nil {
		return nil, err
	}
	assembly, err := AssembleDefault(config)
	if err != nil {
		return nil, err
	}
	return New(
		runner,
		WithRuntime(assembly.Runtime),
		WithCapabilities(assembly.Capabilities...),
	).Build()
}

func validateDefaultPaths(config DefaultConfig) (string, string, error) {
	if config.ContextWindowTokens <= 0 {
		return "", "", errors.New("default context window tokens must be positive")
	}
	if config.WorkspaceRoot == "" || !filepath.IsAbs(config.WorkspaceRoot) {
		return "", "", errors.New("default workspace root must be absolute")
	}
	if config.SessionDir == "" || !filepath.IsAbs(config.SessionDir) {
		return "", "", errors.New("default session directory must be absolute")
	}
	workspace, err := filepath.EvalSymlinks(filepath.Clean(config.WorkspaceRoot))
	if err != nil {
		return "", "", fmt.Errorf("canonicalize default workspace: %w", err)
	}
	info, err := os.Stat(workspace)
	if err != nil {
		return "", "", fmt.Errorf("stat default workspace: %w", err)
	}
	if !info.IsDir() {
		return "", "", errors.New("default workspace root is not a directory")
	}
	sessions, err := canonicalFuturePath(config.SessionDir)
	if err != nil {
		return "", "", fmt.Errorf("canonicalize default session directory: %w", err)
	}
	if pathsOverlap(workspace, sessions) {
		return "", "", errors.New("default workspace and session directory must not overlap")
	}
	return workspace, sessions, nil
}

func canonicalFuturePath(value string) (string, error) {
	value = filepath.Clean(value)
	current := value
	var suffix []string
	for {
		resolved, err := filepath.EvalSymlinks(current)
		if err == nil {
			parts := append([]string{resolved}, suffix...)
			return filepath.Clean(filepath.Join(parts...)), nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", err
		}
		suffix = append([]string{filepath.Base(current)}, suffix...)
		current = parent
	}
}

func pathsOverlap(left, right string) bool {
	return pathWithin(left, right) || pathWithin(right, left)
}

func pathWithin(root, candidate string) bool {
	relative, err := filepath.Rel(root, candidate)
	return err == nil && relative != ".." &&
		!strings.HasPrefix(relative, ".."+string(filepath.Separator)) &&
		!filepath.IsAbs(relative)
}

func defaultSummaryBytes(window int) int {
	value := window / 4
	if value < 256 {
		return 256
	}
	if value > contextpolicy.DefaultStructuredSummaryBytes {
		return contextpolicy.DefaultStructuredSummaryBytes
	}
	return value
}

func artifactReadLimit(config DefaultConfig) int {
	if config.DisableArtifactSpill {
		return artifactcapability.DefaultReadBytes
	}
	threshold := config.SpillThreshold
	if threshold == 0 {
		threshold = spill.DefaultThreshold
	}
	return max(1, min(artifactcapability.DefaultReadBytes, threshold/2))
}
