package mcp

import (
	"context"
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	mcpgo "github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"

	"github.com/regularkevvv/agentic/internal/core"
)

func TestClientConnectListToolsCallToolAndToolsetOverSSE(t *testing.T) {
	mcpServer := server.NewMCPServer(
		"test-server",
		"1.0.0",
		server.WithToolCapabilities(true),
	)
	mcpServer.AddTool(
		mcpgo.NewTool(
			"echo",
			mcpgo.WithDescription("Echo a name"),
			mcpgo.WithString("name"),
		),
		func(ctx context.Context, request mcpgo.CallToolRequest) (*mcpgo.CallToolResult, error) {
			return mcpgo.NewToolResultText(fmt.Sprintf("hello %s", request.GetArguments()["name"])), nil
		},
	)
	mcpServer.AddTool(
		mcpgo.NewTool("fail"),
		func(ctx context.Context, request mcpgo.CallToolRequest) (*mcpgo.CallToolResult, error) {
			return mcpgo.NewToolResultError("boom"), nil
		},
	)

	sseServer := server.NewTestServer(mcpServer)
	defer sseServer.Close()

	client := NewSSEClient("remote-tools", sseServer.URL+"/sse", WithHeaders(map[string]string{"X-Test": "1"}))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("expected repeated Connect to be a no-op, got %v", err)
	}

	tools, err := client.listTools(ctx)
	if err != nil {
		t.Fatalf("listTools: %v", err)
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %#v", tools)
	}

	result, err := client.callTool(ctx, "echo", map[string]any{"name": "Lima"})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if got := extractTextContent(result.Content); got != "hello Lima" {
		t.Fatalf("unexpected tool result %q", got)
	}

	toolset, err := NewToolset(client)
	if err != nil {
		t.Fatalf("NewToolset: %v", err)
	}

	convertedTools, handlers := toolset.ToolsAndHandlers()
	if len(convertedTools) != 2 || len(handlers) != 2 {
		t.Fatalf("unexpected toolset contents: tools=%d handlers=%d", len(convertedTools), len(handlers))
	}

	var echoHandler, failHandler core.ToolHandler
	for _, handler := range handlers {
		switch handler.Name() {
		case "echo":
			echoHandler = handler
		case "fail":
			failHandler = handler
		}
	}
	if echoHandler == nil || failHandler == nil {
		t.Fatalf("expected both handlers to be present, got %#v", handlers)
	}

	out, err := echoHandler.Execute(ctx, map[string]any{"name": "Cusco"}, nil)
	if err != nil {
		t.Fatalf("echo handler execute: %v", err)
	}
	if out != "hello Cusco" {
		t.Fatalf("unexpected echo handler output %#v", out)
	}

	out, err = failHandler.Execute(ctx, map[string]any{}, nil)
	if err == nil || out != "boom" {
		t.Fatalf("expected MCP error result, got output=%#v err=%v", out, err)
	}

	if err := client.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestClientConnectOverHTTPAndStdioFailure(t *testing.T) {
	mcpServer := server.NewMCPServer(
		"http-server",
		"1.0.0",
		server.WithToolCapabilities(true),
	)
	mcpServer.AddTool(
		mcpgo.NewTool("echo_http"),
		func(ctx context.Context, request mcpgo.CallToolRequest) (*mcpgo.CallToolResult, error) {
			return mcpgo.NewToolResultText("ok"), nil
		},
	)

	httpHandler := server.NewStreamableHTTPServer(mcpServer)
	httpServer := httptest.NewServer(httpHandler)
	defer httpServer.Close()

	client := NewHTTPClient("remote-http", httpServer.URL+"/mcp")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("HTTP Connect: %v", err)
	}
	tools, err := client.listTools(ctx)
	if err != nil {
		t.Fatalf("HTTP listTools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "echo_http" {
		t.Fatalf("unexpected HTTP tool list %#v", tools)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("HTTP Close: %v", err)
	}

	stdioClient := NewStdioClient("missing", "/definitely/not/a/real-command", nil)
	if err := stdioClient.Connect(ctx); err == nil {
		t.Fatal("expected stdio connect to fail for a missing command")
	}
}

func TestClientListToolsPaginatesOverHTTP(t *testing.T) {
	mcpServer := server.NewMCPServer(
		"http-server",
		"1.0.0",
		server.WithToolCapabilities(true),
		server.WithPaginationLimit(1),
	)
	for _, name := range []string{"alpha", "beta", "gamma"} {
		mcpServer.AddTool(
			mcpgo.NewTool(name),
			func(ctx context.Context, request mcpgo.CallToolRequest) (*mcpgo.CallToolResult, error) {
				return mcpgo.NewToolResultText("ok"), nil
			},
		)
	}

	httpHandler := server.NewStreamableHTTPServer(mcpServer)
	httpServer := httptest.NewServer(httpHandler)
	defer httpServer.Close()

	client := NewHTTPClient("remote-http", httpServer.URL+"/mcp", WithHeaders(map[string]string{"X-Test": "1"}))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	tools, err := client.listTools(ctx)
	if err != nil {
		t.Fatalf("listTools: %v", err)
	}
	if len(tools) != 3 {
		t.Fatalf("expected all paginated tools, got %#v", tools)
	}

	if err := client.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
