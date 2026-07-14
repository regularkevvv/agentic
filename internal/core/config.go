package core

// ToolConfig holds per-tool configuration.
type ToolConfig struct {
	// MaxRetries overrides the agent's global retry limit for this tool.
	// nil means "use agent's global default".
	MaxRetries *int
}

// ConfigurableToolHandler is an optional interface that ToolHandler implementations
// can implement to provide per-tool configuration (e.g., custom retry limits).
type ConfigurableToolHandler interface {
	ToolHandler
	ToolConfig() *ToolConfig
}
