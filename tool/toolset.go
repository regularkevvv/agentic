package tool

import (
	"context"

	"github.com/regularkevvv/agentic/internal/core"
)

// Toolset is a composable collection of tools and their handlers.
type Toolset interface {
	// ToolsAndHandlers returns all tools and their handlers in this set.
	ToolsAndHandlers() ([]core.Tool, []core.ToolHandler)
}

// FuncToolset is a simple Toolset backed by a slice of tool/handler pairs.
type FuncToolset struct {
	tools    []core.Tool
	handlers []core.ToolHandler
}

// NewToolset creates a new FuncToolset from tool/handler pairs.
func NewToolset() *FuncToolset {
	return &FuncToolset{}
}

// Add adds a tool and handler to the toolset.
func (ts *FuncToolset) Add(tool core.Tool, handler core.ToolHandler) *FuncToolset {
	ts.tools = append(ts.tools, tool)
	ts.handlers = append(ts.handlers, handler)
	return ts
}

// ToolsAndHandlers implements Toolset.
func (ts *FuncToolset) ToolsAndHandlers() ([]core.Tool, []core.ToolHandler) {
	return ts.tools, ts.handlers
}

// CombineToolsets merges multiple toolsets into one.
func CombineToolsets(sets ...Toolset) Toolset {
	result := NewToolset()
	for _, set := range sets {
		tools, handlers := set.ToolsAndHandlers()
		for i := range tools {
			result.Add(tools[i], handlers[i])
		}
	}
	return result
}

// FilterToolset filters a toolset based on a predicate on the tool name.
func FilterToolset(set Toolset, predicate func(toolName string) bool) Toolset {
	tools, handlers := set.ToolsAndHandlers()
	result := NewToolset()
	for i, tool := range tools {
		if predicate(tool.Function.Name) {
			result.Add(tool, handlers[i])
		}
	}
	return result
}

// PrefixToolset adds a prefix to all tool names in a toolset.
// The prefix is joined with "__" (e.g., prefix "math" + tool "add" -> "math__add").
func PrefixToolset(set Toolset, prefix string) Toolset {
	tools, handlers := set.ToolsAndHandlers()
	result := NewToolset()
	for i, tool := range tools {
		prefixedTool := tool
		prefixedTool.Function.Name = prefix + "__" + tool.Function.Name

		// Wrap handler with prefixed name
		wrappedHandler := &renamedHandler{
			inner: handlers[i],
			name:  prefixedTool.Function.Name,
		}
		result.Add(prefixedTool, wrappedHandler)
	}
	return result
}

// renamedHandler wraps a ToolHandler with a different name.
type renamedHandler struct {
	inner core.ToolHandler
	name  string
}

func (h *renamedHandler) Execute(ctx context.Context, input map[string]interface{}, deps any) (interface{}, error) {
	return h.inner.Execute(ctx, input, deps)
}

func (h *renamedHandler) Name() string {
	return h.name
}

// RegisterToolset registers all tools from a Toolset into a ToolRegistry.
func RegisterToolset(registry ToolRegistry, set Toolset) error {
	tools, handlers := set.ToolsAndHandlers()
	for i := range tools {
		if err := registry.Register(tools[i], handlers[i]); err != nil {
			return err
		}
	}
	return nil
}
