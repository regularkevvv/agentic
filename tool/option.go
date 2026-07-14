package tool

import "github.com/regularkevvv/agentic/internal/core"

// ToolOption configures individual tool behavior.
type ToolOption func(*core.ToolConfig)

// WithToolMaxRetries sets the max retry count for a specific tool.
// This overrides the agent's global MaxRetries for this tool only.
func WithToolMaxRetries(n int) ToolOption {
	return func(c *core.ToolConfig) {
		c.MaxRetries = &n
	}
}

// applyToolOptions applies ToolOptions and returns a ToolConfig.
// Returns nil if no options were provided (use global defaults).
func applyToolOptions(opts []ToolOption) *core.ToolConfig {
	if len(opts) == 0 {
		return nil
	}
	cfg := &core.ToolConfig{}
	for _, opt := range opts {
		opt(cfg)
	}
	return cfg
}
