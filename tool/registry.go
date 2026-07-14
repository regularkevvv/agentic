package tool

import (
	"context"
	"fmt"
	"sync"

	"github.com/regularkevvv/agentic/internal/core"
)

// ToolRegistry manages the registration and execution of tools.
type ToolRegistry interface {
	// Register adds a tool and its handler to the registry.
	Register(tool core.Tool, handler core.ToolHandler) error

	// Get retrieves a handler by tool name.
	Get(name string) (core.ToolHandler, bool)

	// Execute runs a single tool call and returns the result.
	Execute(ctx context.Context, toolCall core.ToolUse, deps any) (core.ToolExecutionResult, error)

	// ExecuteBatch runs multiple tool calls and returns all results.
	ExecuteBatch(ctx context.Context, toolCalls []core.ToolUse, deps any) ([]core.ToolExecutionResult, error)

	// Tools returns all registered tool definitions.
	Tools() []core.Tool

	// Has returns true if a tool with the given name is registered.
	Has(name string) bool

	// Count returns the number of registered tools.
	Count() int
}

type defaultRegistry struct {
	tools    map[string]core.Tool
	handlers map[string]core.ToolHandler
	order    []string
	mu       sync.RWMutex
}

// NewRegistry creates a new ToolRegistry.
func NewRegistry() ToolRegistry {
	return &defaultRegistry{
		tools:    make(map[string]core.Tool),
		handlers: make(map[string]core.ToolHandler),
	}
}

func (r *defaultRegistry) Register(tool core.Tool, handler core.ToolHandler) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := tool.Function.Name

	if handler.Name() != name {
		return fmt.Errorf("handler name %q does not match tool name %q", handler.Name(), name)
	}
	if _, exists := r.handlers[name]; exists {
		return fmt.Errorf("tool %q already registered", name)
	}

	r.tools[name] = tool
	r.handlers[name] = handler
	r.order = append(r.order, name)
	return nil
}

func (r *defaultRegistry) Get(name string) (core.ToolHandler, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	handler, ok := r.handlers[name]
	return handler, ok
}

func (r *defaultRegistry) Execute(
	ctx context.Context,
	toolCall core.ToolUse,
	deps any,
) (core.ToolExecutionResult, error) {
	handler, ok := r.Get(toolCall.Name)
	if !ok {
		return core.ToolExecutionResult{
			ToolUseID: toolCall.ID,
			ToolName:  toolCall.Name,
			Content:   fmt.Sprintf("Unknown tool: %s", toolCall.Name),
			IsError:   true,
			Error:     fmt.Errorf("unknown tool: %s", toolCall.Name),
		}, nil
	}

	r.mu.RLock()
	tool := r.tools[toolCall.Name]
	r.mu.RUnlock()
	if err := validateToolInput(toolCall.Input, tool.Function.Parameters); err != nil {
		retryErr := &core.ModelRetry{Message: fmt.Sprintf("Invalid arguments for tool %q: %s", toolCall.Name, err)}
		return core.ToolExecutionResult{
			ToolUseID: toolCall.ID,
			ToolName:  toolCall.Name,
			Content:   retryErr.Error(),
			IsError:   true,
			Error:     retryErr,
		}, nil
	}

	result, err := handler.Execute(ctx, toolCall.Input, deps)
	if err != nil {
		return core.ToolExecutionResult{
			ToolUseID: toolCall.ID,
			ToolName:  toolCall.Name,
			Content:   err.Error(),
			IsError:   true,
			Error:     err,
		}, nil
	}

	return core.ToolExecutionResult{
		ToolUseID: toolCall.ID,
		ToolName:  toolCall.Name,
		Content:   result,
		IsError:   false,
	}, nil
}

func (r *defaultRegistry) ExecuteBatch(
	ctx context.Context,
	toolCalls []core.ToolUse,
	deps any,
) ([]core.ToolExecutionResult, error) {
	results := make([]core.ToolExecutionResult, len(toolCalls))
	errs := make([]error, len(toolCalls))
	var wg sync.WaitGroup
	wg.Add(len(toolCalls))
	for i, toolCall := range toolCalls {
		go func() {
			defer wg.Done()
			result, err := r.Execute(ctx, toolCall, deps)
			results[i] = result
			errs[i] = err
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("execute tool %q: %w", toolCalls[i].Name, err)
		}
	}
	return results, nil
}

func (r *defaultRegistry) Tools() []core.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tools := make([]core.Tool, 0, len(r.order))
	for _, name := range r.order {
		tools = append(tools, r.tools[name])
	}
	return tools
}

func (r *defaultRegistry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.handlers[name]
	return ok
}

func (r *defaultRegistry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.handlers)
}
