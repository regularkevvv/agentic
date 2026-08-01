// Package codemode wraps explicitly selected tools behind one host-managed
// execution capability. Language semantics are supplied by an Executor port;
// this package contains no interpreter.
package codemode

import "context"

type Checkpoint []byte

type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type Request struct {
	Code  string `json:"code"`
	Tools []Tool `json:"tools"`
}

type Call struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

type CallResult struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Content any    `json:"content,omitempty"`
	IsError bool   `json:"is_error,omitempty"`
}

type Step struct {
	Checkpoint Checkpoint `json:"checkpoint,omitempty"`
	Calls      []Call     `json:"calls,omitempty"`
	Output     any        `json:"output,omitempty"`
	Stdout     string     `json:"stdout,omitempty"`
	Done       bool       `json:"done"`
}

type Result struct {
	Output any    `json:"output,omitempty"`
	Stdout string `json:"stdout,omitempty"`
}

type Executor interface {
	Start(context.Context, Request) (Step, error)
	Resume(context.Context, Checkpoint, []CallResult) (Step, error)
}

type ExecutorFunc struct {
	StartFunc  func(context.Context, Request) (Step, error)
	ResumeFunc func(context.Context, Checkpoint, []CallResult) (Step, error)
}

func (f ExecutorFunc) Start(ctx context.Context, request Request) (Step, error) {
	return f.StartFunc(ctx, request)
}

func (f ExecutorFunc) Resume(ctx context.Context, checkpoint Checkpoint, results []CallResult) (Step, error) {
	return f.ResumeFunc(ctx, checkpoint, results)
}
