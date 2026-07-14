package mcp

import (
	"context"
	"fmt"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	mcptypes "github.com/mark3labs/mcp-go/mcp"
)

// Client wraps an MCP server connection for use with Agentic agents.
type Client struct {
	inner     *mcpclient.Client
	name      string
	connected bool
	stdioCmd  string
	stdioArgs []string
	stdioEnv  []string
	httpURL   string
	httpHdrs  map[string]string
	kind      clientKind
}

type clientKind int

const (
	kindStdio clientKind = iota
	kindSSE
	kindHTTP
)

// Option configures an MCP Client.
type Option func(*clientConfig)

type clientConfig struct {
	env     map[string]string
	headers map[string]string
}

// WithEnv sets environment variables for stdio transport.
func WithEnv(env map[string]string) Option {
	return func(c *clientConfig) { c.env = env }
}

// WithHeaders sets HTTP headers for SSE and HTTP transports.
func WithHeaders(headers map[string]string) Option {
	return func(c *clientConfig) { c.headers = headers }
}

// NewStdioClient creates an MCP client that communicates over stdio.
// The command and args specify the MCP server process to launch.
//
// Example:
//
//	client := mcp.NewStdioClient("filesystem", "npx",
//	    []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
//	)
func NewStdioClient(name string, command string, args []string, opts ...Option) *Client {
	cfg := &clientConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	var env []string
	for k, v := range cfg.env {
		env = append(env, k+"="+v)
	}

	return &Client{
		name:      name,
		kind:      kindStdio,
		stdioCmd:  command,
		stdioArgs: args,
		stdioEnv:  env,
	}
}

// NewSSEClient creates an MCP client that communicates over HTTP/SSE.
//
// Example:
//
//	client := mcp.NewSSEClient("remote-tools", "http://localhost:3000/sse")
func NewSSEClient(name string, url string, opts ...Option) *Client {
	cfg := &clientConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	return &Client{
		name:     name,
		kind:     kindSSE,
		httpURL:  url,
		httpHdrs: cfg.headers,
	}
}

// NewHTTPClient creates an MCP client that communicates over Streamable HTTP.
// This is the newer MCP transport that uses standard HTTP requests with
// optional streaming responses.
//
// Example:
//
//	client := mcp.NewHTTPClient("remote-tools", "http://localhost:3000/mcp")
func NewHTTPClient(name string, url string, opts ...Option) *Client {
	cfg := &clientConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	return &Client{
		name:     name,
		kind:     kindHTTP,
		httpURL:  url,
		httpHdrs: cfg.headers,
	}
}

// Connect establishes the connection to the MCP server and performs the
// initialization handshake.
func (c *Client) Connect(ctx context.Context) error {
	if c.connected {
		return nil
	}

	var err error
	switch c.kind {
	case kindStdio:
		c.inner, err = mcpclient.NewStdioMCPClient(c.stdioCmd, c.stdioEnv, c.stdioArgs...)
		if err != nil {
			return fmt.Errorf("mcp stdio connect: %w", err)
		}

	case kindSSE:
		var transportOpts []transport.ClientOption
		if len(c.httpHdrs) > 0 {
			transportOpts = append(transportOpts, transport.WithHeaders(c.httpHdrs))
		}
		c.inner, err = mcpclient.NewSSEMCPClient(c.httpURL, transportOpts...)
		if err != nil {
			return fmt.Errorf("mcp sse connect: %w", err)
		}

	case kindHTTP:
		var httpOpts []transport.StreamableHTTPCOption
		if len(c.httpHdrs) > 0 {
			httpOpts = append(httpOpts, transport.WithHTTPHeaders(c.httpHdrs))
		}
		c.inner, err = mcpclient.NewStreamableHttpClient(c.httpURL, httpOpts...)
		if err != nil {
			return fmt.Errorf("mcp http connect: %w", err)
		}
	}

	if err = c.inner.Start(ctx); err != nil {
		_ = c.inner.Close()
		return fmt.Errorf("mcp start: %w", err)
	}

	// Perform initialization handshake
	initReq := mcptypes.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcptypes.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcptypes.Implementation{
		Name:    "agentic",
		Version: "0.1.0",
	}
	initReq.Params.Capabilities = mcptypes.ClientCapabilities{}

	_, err = c.inner.Initialize(ctx, initReq)
	if err != nil {
		_ = c.inner.Close()
		return fmt.Errorf("mcp initialize: %w", err)
	}

	c.connected = true
	return nil
}

// Close shuts down the MCP server connection.
func (c *Client) Close() error {
	if c.inner == nil {
		return nil
	}
	c.connected = false
	return c.inner.Close()
}

// Name returns the server name.
func (c *Client) Name() string {
	return c.name
}

// listTools fetches the tool list from the MCP server.
func (c *Client) listTools(ctx context.Context) ([]mcptypes.Tool, error) {
	if !c.connected {
		return nil, fmt.Errorf("mcp client not connected (call Connect first)")
	}

	var allTools []mcptypes.Tool
	req := mcptypes.ListToolsRequest{}

	for {
		result, err := c.inner.ListTools(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("mcp list tools: %w", err)
		}

		allTools = append(allTools, result.Tools...)

		if result.NextCursor == "" {
			break
		}
		req.Params.Cursor = result.NextCursor
	}

	return allTools, nil
}

// callTool invokes a tool on the MCP server.
func (c *Client) callTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcptypes.CallToolResult, error) {
	if !c.connected {
		return nil, fmt.Errorf("mcp client not connected (call Connect first)")
	}

	req := mcptypes.CallToolRequest{}
	req.Params.Name = name
	req.Params.Arguments = arguments

	return c.inner.CallTool(ctx, req)
}
