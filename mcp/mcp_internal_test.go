package mcp

import (
	"context"
	"strings"
	"testing"

	mcptypes "github.com/mark3labs/mcp-go/mcp"
)

func TestClientConstructorsAndHelpers(t *testing.T) {
	stdio := NewStdioClient("filesystem", "npx", []string{"server"}, WithEnv(map[string]string{"A": "1"}))
	if stdio.Name() != "filesystem" {
		t.Fatalf("unexpected stdio client name %q", stdio.Name())
	}
	if stdio.kind != kindStdio || stdio.stdioCmd != "npx" || len(stdio.stdioArgs) != 1 || len(stdio.stdioEnv) != 1 {
		t.Fatalf("unexpected stdio client config %#v", stdio)
	}

	sse := NewSSEClient("remote", "http://localhost/sse", WithHeaders(map[string]string{"Authorization": "Bearer token"}))
	if sse.kind != kindSSE || sse.httpURL != "http://localhost/sse" || sse.httpHdrs["Authorization"] != "Bearer token" {
		t.Fatalf("unexpected SSE client config %#v", sse)
	}

	http := NewHTTPClient("remote-http", "http://localhost/mcp", WithHeaders(map[string]string{"X-Test": "1"}))
	if http.kind != kindHTTP || http.httpURL != "http://localhost/mcp" || http.httpHdrs["X-Test"] != "1" {
		t.Fatalf("unexpected HTTP client config %#v", http)
	}

	if err := http.Close(); err != nil {
		t.Fatalf("Close on disconnected client should be nil, got %v", err)
	}
}

func TestToolsetConversionAndErrorPaths(t *testing.T) {
	client := NewHTTPClient("remote-http", "http://localhost/mcp")

	if _, err := NewToolset(client); err == nil || !strings.Contains(err.Error(), "call Connect first") {
		t.Fatalf("expected disconnected client error from NewToolset, got %v", err)
	}
	if _, err := NewToolsetWithContext(context.Background(), client); err == nil || !strings.Contains(err.Error(), "call Connect first") {
		t.Fatalf("expected disconnected client error from NewToolsetWithContext, got %v", err)
	}

	mcpTool := mcptypes.Tool{
		Name:        "lookup_weather",
		Description: "Lookup weather",
		InputSchema: mcptypes.ToolInputSchema{
			Type: "object",
			Properties: map[string]any{
				"city": map[string]any{"type": "string"},
			},
			Required: []string{"city"},
		},
	}

	tool, handler := convertMCPTool(client, mcpTool)
	if tool.Function.Name != "lookup_weather" {
		t.Fatalf("unexpected converted tool name %q", tool.Function.Name)
	}
	if handler.Name() != "lookup_weather" {
		t.Fatalf("unexpected handler name %q", handler.Name())
	}

	text := extractTextContent([]mcptypes.Content{
		mcptypes.NewTextContent("hello"),
		&mcptypes.TextContent{Type: "text", Text: "world"},
	})
	if text != "hello\nworld" {
		t.Fatalf("unexpected extracted text %q", text)
	}

	withPlaceholder := extractTextContent([]mcptypes.Content{
		mcptypes.NewTextContent("hello"),
		&mcptypes.ImageContent{Type: "image", MIMEType: "image/png", Data: "AQID"},
	})
	if !strings.Contains(withPlaceholder, "[*mcp.ImageContent content]") {
		t.Fatalf("expected placeholder for non-text content, got %q", withPlaceholder)
	}

	if _, err := handler.Execute(context.Background(), map[string]interface{}{"city": "Lima"}, nil); err == nil || !strings.Contains(err.Error(), `mcp tool "lookup_weather"`) {
		t.Fatalf("expected wrapped MCP execution error, got %v", err)
	}
}
