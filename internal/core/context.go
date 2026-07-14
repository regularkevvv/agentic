package core

import "context"

// RunContext provides context and dependencies to dependency-aware callbacks.
//
// D is the exact dependency type selected by the caller. The framework does
// not add a pointer; applications that want pointer semantics use a pointer as
// D (for example, RunContext[*MyDeps]).
//
//	type MyDeps struct {
//	    DB     *sql.DB
//	    Logger *slog.Logger
//	    Cache  map[string]string
//	}
type RunContext[D any] struct {
	// Ctx is the standard context for cancellation and timeouts.
	Ctx context.Context

	// Deps contains the exact dependency value supplied for this run.
	Deps D

	// Retry is the current retry count for this tool execution.
	Retry int

	// ToolName is the name of the currently executing tool.
	ToolName string
}
