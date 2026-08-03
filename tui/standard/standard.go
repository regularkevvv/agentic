// Package standard owns the optional opinionated assembly from an explicit
// provider profile to Agentic, Harness Default, and the reusable TUI adapter.
package standard

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	agentic "github.com/regularkevvv/agentic"
	harnesscore "github.com/regularkevvv/agentic/harness"
	"github.com/regularkevvv/agentic/harness/permission"
	"github.com/regularkevvv/agentic/provider/anthropic"
	"github.com/regularkevvv/agentic/provider/openai"
	"github.com/regularkevvv/agentic/provider/openrouter"
	uit "github.com/regularkevvv/agentic/tui"
	harnessui "github.com/regularkevvv/agentic/tui/adapter/harness"
	appconfig "github.com/regularkevvv/agentic/tui/config"
)

type ModelConfig struct {
	Model               string
	ContextWindowTokens int
}

type ProviderFactory interface {
	ID() string
	New(context.Context, ModelConfig) (agentic.Model, error)
}

type FactoryFunc struct {
	Name   string
	Create func(context.Context, ModelConfig) (agentic.Model, error)
}

func (f FactoryFunc) ID() string { return f.Name }
func (f FactoryFunc) New(ctx context.Context, config ModelConfig) (agentic.Model, error) {
	if f.Create == nil {
		return nil, errors.New("provider factory has no constructor")
	}
	return f.Create(ctx, config)
}

type Registry struct {
	mu        sync.RWMutex
	factories map[string]ProviderFactory
}

func NewRegistry(factories ...ProviderFactory) (*Registry, error) {
	result := &Registry{factories: make(map[string]ProviderFactory)}
	for _, factory := range factories {
		if err := result.Register(factory); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func (r *Registry) Register(factory ProviderFactory) error {
	if r == nil || factory == nil || factory.ID() == "" {
		return errors.New("provider factory and ID are required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.factories[factory.ID()]; exists {
		return fmt.Errorf("provider %q is already registered", factory.ID())
	}
	r.factories[factory.ID()] = factory
	return nil
}

func (r *Registry) Get(id string) (ProviderFactory, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	value, ok := r.factories[id]
	return value, ok
}

func (r *Registry) IDs() []string {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]string, 0, len(r.factories))
	for id := range r.factories {
		result = append(result, id)
	}
	sort.Strings(result)
	return result
}

type Config struct {
	ProfileName          string
	Provider             string
	Model                string
	ContextWindowTokens  int
	SystemPromptFile     string
	WorkspaceRoot        string
	SessionDirectory     string
	Permission           string
	PermissionPolicy     *permission.Policy
	PromptCacheRetention agentic.PromptCacheRetention
}

func FromResolved(value appconfig.Resolved) Config {
	return Config{
		ProfileName: value.ProfileName, Provider: value.Provider, Model: value.Model,
		ContextWindowTokens: value.ContextWindowTokens, SystemPromptFile: value.SystemPromptFile,
		WorkspaceRoot: value.WorkspaceRoot, SessionDirectory: value.SessionDirectory,
		Permission: value.Permission, PromptCacheRetention: agentic.PromptCacheShort,
	}
}

type Assembly struct {
	Host    uit.Host
	Runtime *harnesscore.Harness[string]
}

func Build(ctx context.Context, registry *Registry, config Config) (Assembly, error) {
	if registry == nil {
		return Assembly{}, errors.New("provider registry is required")
	}
	if config.Provider == "" || config.Model == "" || config.ContextWindowTokens <= 0 {
		return Assembly{}, errors.New("provider, model, and positive context window are required")
	}
	if config.SystemPromptFile == "" || config.WorkspaceRoot == "" || config.SessionDirectory == "" {
		return Assembly{}, errors.New("system prompt file, workspace root, and session directory are required")
	}
	if !filepath.IsAbs(config.WorkspaceRoot) || !filepath.IsAbs(config.SessionDirectory) || !filepath.IsAbs(config.SystemPromptFile) {
		return Assembly{}, errors.New("standard assembly paths must be absolute")
	}
	info, err := os.Stat(config.WorkspaceRoot)
	if err != nil {
		return Assembly{}, fmt.Errorf("workspace root is not an accessible directory: %w", err)
	}
	if !info.IsDir() {
		return Assembly{}, errors.New("workspace root is not a directory")
	}
	systemPrompt, err := os.ReadFile(config.SystemPromptFile)
	if err != nil {
		return Assembly{}, fmt.Errorf("read system prompt: %w", err)
	}
	if len(systemPrompt) == 0 {
		return Assembly{}, errors.New("system prompt file is empty")
	}
	factory, ok := registry.Get(config.Provider)
	if !ok {
		return Assembly{}, fmt.Errorf("provider %q is not registered", config.Provider)
	}
	model, err := factory.New(ctx, ModelConfig{Model: config.Model, ContextWindowTokens: config.ContextWindowTokens})
	if err != nil {
		return Assembly{}, fmt.Errorf("construct provider %s: %w", config.Provider, err)
	}
	if model == nil {
		return Assembly{}, fmt.Errorf("provider %q returned a nil model", config.Provider)
	}
	policy, err := resolvePermission(config.Permission, config.PermissionPolicy)
	if err != nil {
		return Assembly{}, err
	}
	retention := config.PromptCacheRetention
	if retention == "" {
		retention = agentic.PromptCacheShort
	}
	runner := agentic.NewAgent(string(systemPrompt), model)
	runtime, err := harnesscore.Default[string](runner, harnesscore.DefaultConfig{
		WorkspaceRoot: config.WorkspaceRoot, SessionDir: config.SessionDirectory,
		ContextWindowTokens: config.ContextWindowTokens, PermissionPolicy: policy,
		PromptCacheRetention: retention, ModelStreaming: true,
	})
	if err != nil {
		return Assembly{}, err
	}
	host, err := harnessui.New(runtime, harnessui.WithProfileLabel(config.ProfileName), harnessui.WithWorkspace(config.WorkspaceRoot))
	if err != nil {
		return Assembly{}, err
	}
	return Assembly{Host: host, Runtime: runtime}, nil
}

func resolvePermission(name string, custom *permission.Policy) (*permission.Policy, error) {
	switch name {
	case "", "workspace-write":
		return permission.WorkspaceWrite(), nil
	case "read-only":
		return permission.ReadOnly(), nil
	case "custom":
		if custom == nil {
			return nil, errors.New("custom permission requires an application-supplied policy")
		}
		return custom, nil
	default:
		return nil, fmt.Errorf("unsupported permission policy %q", name)
	}
}

type EnvironmentLookup func(string) (string, bool)

func BuiltinFactories(lookup EnvironmentLookup) []ProviderFactory {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	credential := func(providerID, variable string, create func(string, string) (agentic.Model, error)) ProviderFactory {
		return FactoryFunc{Name: providerID, Create: func(_ context.Context, config ModelConfig) (agentic.Model, error) {
			key, ok := lookup(variable)
			if !ok || key == "" {
				return nil, fmt.Errorf("credential %s is not available", variable)
			}
			return create(config.Model, key)
		}}
	}
	return []ProviderFactory{
		credential("anthropic", "ANTHROPIC_API_KEY", func(model, key string) (agentic.Model, error) {
			return anthropic.New(model, anthropic.WithAPIKey(key))
		}),
		credential("openai", "OPENAI_API_KEY", func(model, key string) (agentic.Model, error) {
			return openai.New(model, openai.WithAPIKey(key))
		}),
		credential("openrouter", "OPENROUTER_API_KEY", func(model, key string) (agentic.Model, error) {
			return openrouter.New(model, openrouter.WithAPIKey(key))
		}),
	}
}
