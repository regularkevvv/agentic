package codemode

import (
	"errors"
	"fmt"
	"regexp"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/capability"
)

const (
	DeferralKind = "harness.codemode.v1"
	defaultID    = "codemode"
	defaultTool  = "run_code"
)

var identifierPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

type Limits struct {
	MaxCodeBytes       int
	MaxSteps           int
	MaxCallsPerStep    int
	MaxCheckpointBytes int
	MaxValueBytes      int
	MaxStdoutBytes     int
}

func (l Limits) withDefaults() Limits {
	if l.MaxCodeBytes == 0 {
		l.MaxCodeBytes = 64 << 10
	}
	if l.MaxSteps == 0 {
		l.MaxSteps = 32
	}
	if l.MaxCallsPerStep == 0 {
		l.MaxCallsPerStep = 32
	}
	if l.MaxCheckpointBytes == 0 {
		l.MaxCheckpointBytes = 1 << 20
	}
	if l.MaxValueBytes == 0 {
		l.MaxValueBytes = 1 << 20
	}
	if l.MaxStdoutBytes == 0 {
		l.MaxStdoutBytes = 1 << 20
	}
	return l
}

func (l Limits) validate() error {
	if l.MaxCodeBytes <= 0 || l.MaxSteps <= 0 || l.MaxCallsPerStep <= 0 ||
		l.MaxCheckpointBytes <= 0 || l.MaxValueBytes <= 0 || l.MaxStdoutBytes <= 0 {
		return errors.New("codemode limits must be positive")
	}
	return nil
}

type Config struct {
	ID            string
	Order         capability.Ordering
	ToolName      string
	SelectedTools []string
	Executor      Executor
	Limits        Limits
}

type Capability struct {
	config Config
}

func New(config Config) *Capability {
	if config.ID == "" {
		config.ID = defaultID
	}
	if config.ToolName == "" {
		config.ToolName = defaultTool
	}
	config.SelectedTools = append([]string(nil), config.SelectedTools...)
	config.Limits = config.Limits.withDefaults()
	return &Capability{config: config}
}

func (c *Capability) ID() string {
	if c == nil {
		return defaultID
	}
	return c.config.ID
}

func (c *Capability) Ordering() capability.Ordering {
	if c == nil {
		return capability.Ordering{}
	}
	return c.config.Order
}

func (c *Capability) Register(registry *capability.Registry) error {
	if c == nil || c.config.Executor == nil {
		return errors.New("codemode executor is required")
	}
	if !identifierPattern.MatchString(c.config.ToolName) {
		return fmt.Errorf("invalid codemode tool name %q", c.config.ToolName)
	}
	if err := c.config.Limits.validate(); err != nil {
		return err
	}
	if len(c.config.SelectedTools) == 0 {
		return errors.New("codemode selected tools are required")
	}
	if registry.HasTool(c.config.ToolName) {
		return fmt.Errorf("codemode tool name %q is already registered", c.config.ToolName)
	}
	for _, name := range c.config.SelectedTools {
		if !identifierPattern.MatchString(name) {
			return fmt.Errorf("invalid selected tool name %q", name)
		}
		if registry.IsDelegationTool(name) {
			return fmt.Errorf("codemode cannot select delegation tool %q", name)
		}
	}
	selected, err := registry.TakeToolset(c.config.SelectedTools...)
	if err != nil {
		return err
	}
	definitions, handlers := selected.ToolsAndHandlers()
	toolRegistry := agentic.NewRegistry()
	catalog := make([]Tool, len(definitions))
	for index, definition := range definitions {
		handler := handlers[index]
		if handler == nil || handler.Name() != definition.Function.Name {
			return fmt.Errorf("selected tool %q has an invalid handler", definition.Function.Name)
		}
		if marker, ok := handler.(agentic.SuspendableToolHandler); ok && marker.MaySuspendToolExecution() {
			return fmt.Errorf("codemode cannot select suspendable tool %q", definition.Function.Name)
		}
		if err := toolRegistry.Register(definition, handler); err != nil {
			return fmt.Errorf("register selected tool %q: %w", definition.Function.Name, err)
		}
		catalog[index] = Tool{
			Name:        definition.Function.Name,
			Description: definition.Function.Description,
			Parameters:  cloneMap(definition.Function.Parameters),
		}
	}
	handler := &runCodeHandler{
		name:     c.config.ToolName,
		executor: c.config.Executor,
		limits:   c.config.Limits,
		registry: toolRegistry,
		catalog:  catalog,
	}
	tool := agentic.Tool{
		Type: agentic.ToolTypeFunction,
		Function: agentic.Function{
			Name:        c.config.ToolName,
			Description: "Execute code that may call the selected host tools.",
			Parameters: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"code": map[string]any{"type": "string"}},
				"required":             []string{"code"},
				"additionalProperties": false,
			},
		},
	}
	toolset := agentic.NewToolset().Add(tool, handler)
	if err := registry.AddToolset(toolset); err != nil {
		return err
	}
	if err := registry.AddResumePlanner(DeferralKind, resumePlanner{toolName: c.config.ToolName, limits: c.config.Limits}); err != nil {
		return err
	}
	return nil
}
