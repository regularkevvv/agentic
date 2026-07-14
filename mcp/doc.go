// Package mcp provides Model Context Protocol (MCP) client integration for Agentic.
//
// MCP allows agents to use tools provided by external MCP servers. Three transport
// types are supported:
//
//   - Stdio: launches a local process and communicates over stdin/stdout
//   - SSE: connects to a server using Server-Sent Events
//   - HTTP: connects using Streamable HTTP (the newer MCP transport)
//
// Tools discovered from MCP servers are automatically converted to Agentic
// tools and can be added to agents using AddToolset.
//
// Example usage:
//
//	// Connect to an MCP server over stdio
//	mcpClient := mcp.NewStdioClient("filesystem", "npx",
//	    []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
//	)
//	if err := mcpClient.Connect(ctx); err != nil {
//	    log.Fatal(err)
//	}
//	defer mcpClient.Close()
//
//	// Create a toolset from the server's tools
//	ts, err := mcp.NewToolset(mcpClient)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Add to agent
//	agent.AddToolset(ts)
package mcp
