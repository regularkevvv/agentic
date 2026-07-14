package mcp

import (
	"context"
	"fmt"
	"strings"

	"github.com/regularkevvv/agentic/internal/core"

	mcptypes "github.com/mark3labs/mcp-go/mcp"
)

// Toolset implements the tools.Toolset interface using tools from an MCP server.
// Use NewToolset to create one from a connected Client.
type Toolset struct {
	client   *Client
	tools    []core.Tool
	handlers []core.ToolHandler
}

// NewToolset creates a Toolset by discovering tools from the MCP server.
// The client must be connected (via Connect) before calling this.
func NewToolset(client *Client) (*Toolset, error) {
	return NewToolsetWithContext(context.Background(), client)
}

// NewToolsetWithContext creates a Toolset using the provided context.
func NewToolsetWithContext(ctx context.Context, client *Client) (*Toolset, error) {
	mcpTools, err := client.listTools(ctx)
	if err != nil {
		return nil, err
	}

	ts := &Toolset{
		client:   client,
		tools:    make([]core.Tool, 0, len(mcpTools)),
		handlers: make([]core.ToolHandler, 0, len(mcpTools)),
	}

	for _, mt := range mcpTools {
		tool, handler := convertMCPTool(client, mt)
		ts.tools = append(ts.tools, tool)
		ts.handlers = append(ts.handlers, handler)
	}

	return ts, nil
}

// ToolsAndHandlers implements tools.Toolset.
func (ts *Toolset) ToolsAndHandlers() ([]core.Tool, []core.ToolHandler) {
	return ts.tools, ts.handlers
}

// convertMCPTool converts an MCP tool definition to an Agentic Tool and ToolHandler.
func convertMCPTool(client *Client, mt mcptypes.Tool) (core.Tool, core.ToolHandler) {
	// Build JSON Schema parameters from MCP InputSchema
	params := make(map[string]interface{})
	params["type"] = mt.InputSchema.Type
	if mt.InputSchema.Properties != nil {
		params["properties"] = mt.InputSchema.Properties
	}
	if len(mt.InputSchema.Required) > 0 {
		params["required"] = mt.InputSchema.Required
	}

	tool := core.Tool{
		Type: core.ToolTypeFunction,
		Function: core.Function{
			Name:        mt.Name,
			Description: mt.Description,
			Parameters:  params,
		},
	}

	handler := &mcpToolHandler{
		client:   client,
		toolName: mt.Name,
	}

	return tool, handler
}

// mcpToolHandler implements core.ToolHandler for an MCP-backed tool.
type mcpToolHandler struct {
	client   *Client
	toolName string
}

// Execute calls the tool on the MCP server.
func (h *mcpToolHandler) Execute(ctx context.Context, input map[string]interface{}, deps any) (interface{}, error) {
	result, err := h.client.callTool(ctx, h.toolName, input)
	if err != nil {
		return nil, fmt.Errorf("mcp tool %q: %w", h.toolName, err)
	}

	if result.IsError {
		return extractTextContent(result.Content), fmt.Errorf("mcp tool %q returned error: %s", h.toolName, extractTextContent(result.Content))
	}

	return extractTextContent(result.Content), nil
}

// Name returns the tool name.
func (h *mcpToolHandler) Name() string {
	return h.toolName
}

// extractTextContent concatenates text content from MCP tool results.
func extractTextContent(content []mcptypes.Content) string {
	var parts []string
	for _, c := range content {
		switch v := c.(type) {
		case mcptypes.TextContent:
			parts = append(parts, v.Text)
		case *mcptypes.TextContent:
			parts = append(parts, v.Text)
		default:
			// For non-text content, include a placeholder
			parts = append(parts, fmt.Sprintf("[%T content]", c))
		}
	}
	return strings.Join(parts, "\n")
}
