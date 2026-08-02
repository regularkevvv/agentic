// Package gomonty adapts the checkpointed codemode Executor port to Monty.
//
// The adapter depends on github.com/regularkevvv/gomonty, but the codemode
// core remains interpreter-neutral. Applications must explicitly prepare the
// verified native runtime before execution. Importing this package, building a
// program, and ordinary harness startup never download or compile native code.
//
// Monty's worker subprocess provides crash isolation and timeout enforcement;
// it is not an OS sandbox. Calls to selected host tools currently require
// keyword arguments so their inputs preserve the codemode object contract.
// Expression results are returned, but the current low-level snapshot API does
// not expose captured print chunks across durable resume, so Step.Stdout is
// empty.
package gomonty

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"slices"

	monty "github.com/regularkevvv/gomonty"

	"github.com/regularkevvv/agentic/harness/codemode"
)

const checkpointVersion = 1

// Config controls Monty compilation and execution limits. Native runtime
// acquisition remains an explicit application concern through monty.Prepare
// or the gomonty prepare command.
type Config struct {
	CompileOptions monty.CompileOptions
	Limits         *monty.ResourceLimits
}

// Executor implements codemode.Executor with Monty's serializable low-level
// snapshots. It contains no session state, so checkpoints may be resumed by a
// fresh Executor after process restart.
type Executor struct {
	compile monty.CompileOptions
	limits  *monty.ResourceLimits
	engine  engine
}

// New constructs an optional Monty adapter without loading, downloading, or
// building native code.
func New(config Config) *Executor {
	compile := config.CompileOptions
	compile.Inputs = slices.Clone(compile.Inputs)
	if compile.ScriptName == "" {
		compile.ScriptName = "agentic-codemode.py"
	}
	if compile.AssertMessageAnnotations != nil {
		value := *compile.AssertMessageAnnotations
		compile.AssertMessageAnnotations = &value
	}
	var limits *monty.ResourceLimits
	if config.Limits != nil {
		value := *config.Limits
		limits = &value
	}
	return &Executor{compile: compile, limits: limits, engine: newMontyEngine()}
}

type engine interface {
	Start(context.Context, string, monty.CompileOptions, *monty.ResourceLimits) (*frontier, error)
	Restore([]byte) (*frontier, error)
}

type frontierKind uint8

const (
	frontierComplete frontierKind = iota + 1
	frontierLookup
	frontierCall
	frontierFuture
)

type callFrontier struct {
	ID         string
	Name       string
	Input      map[string]any
	Positional int
	OS         bool
	Method     bool
}

type frontier struct {
	kind          frontierKind
	output        any
	lookup        string
	call          callFrontier
	resolveLookup func(context.Context, bool) (*frontier, error)
	dump          func() ([]byte, error)
	resumeCall    func(context.Context, codemode.CallResult) (*frontier, error)
	abandon       func() error
}

type durableCheckpoint struct {
	Version  int      `json:"version"`
	Snapshot []byte   `json:"snapshot"`
	Tools    []string `json:"tools"`
	CallID   string   `json:"call_id"`
	CallName string   `json:"call_name"`
}

func (e *Executor) Start(ctx context.Context, request codemode.Request) (codemode.Step, error) {
	if e == nil {
		return codemode.Step{}, errors.New("gomonty executor is nil")
	}
	tools, err := toolNames(request.Tools)
	if err != nil {
		return codemode.Step{}, err
	}
	progress, err := e.engine.Start(ctx, request.Code, e.compile, cloneLimits(e.limits))
	if err != nil {
		return codemode.Step{}, err
	}
	return advance(ctx, progress, tools)
}

func (e *Executor) Resume(
	ctx context.Context,
	checkpoint codemode.Checkpoint,
	results []codemode.CallResult,
) (codemode.Step, error) {
	if e == nil {
		return codemode.Step{}, errors.New("gomonty executor is nil")
	}
	state, err := decodeCheckpoint(checkpoint)
	if err != nil {
		return codemode.Step{}, err
	}
	if len(results) != 1 {
		return codemode.Step{}, fmt.Errorf("gomonty checkpoint requires exactly one result, got %d", len(results))
	}
	result := results[0]
	if result.ID != state.CallID || result.Name != state.CallName {
		return codemode.Step{}, fmt.Errorf(
			"gomonty result identity %q/%q does not match checkpoint %q/%q",
			result.ID, result.Name, state.CallID, state.CallName,
		)
	}
	progress, err := e.engine.Restore(state.Snapshot)
	if err != nil {
		return codemode.Step{}, fmt.Errorf("restore Monty checkpoint: %w", err)
	}
	if progress == nil || progress.kind != frontierCall {
		_ = abandon(progress)
		return codemode.Step{}, errors.New("restored Monty checkpoint is not a function call")
	}
	if progress.call.OS || progress.call.Method || progress.call.Name != state.CallName || progress.call.ID != state.CallID {
		_ = abandon(progress)
		return codemode.Step{}, errors.New("restored Monty call frontier differs from its checkpoint")
	}
	if progress.resumeCall == nil {
		_ = abandon(progress)
		return codemode.Step{}, errors.New("restored Monty call cannot resume")
	}
	resumed, err := progress.resumeCall(ctx, result)
	if err != nil {
		return codemode.Step{}, fmt.Errorf("resume Monty call %q: %w", state.CallName, err)
	}
	return advance(ctx, resumed, state.Tools)
}

func advance(ctx context.Context, progress *frontier, tools []string) (codemode.Step, error) {
	selected := make(map[string]struct{}, len(tools))
	for _, name := range tools {
		selected[name] = struct{}{}
	}
	for {
		if err := ctx.Err(); err != nil {
			_ = abandon(progress)
			return codemode.Step{}, err
		}
		if progress == nil {
			return codemode.Step{}, errors.New("monty produced nil progress")
		}
		switch progress.kind {
		case frontierComplete:
			return codemode.Step{Done: true, Output: progress.output}, nil
		case frontierLookup:
			if progress.resolveLookup == nil {
				_ = abandon(progress)
				return codemode.Step{}, errors.New("monty name lookup cannot resume")
			}
			_, selectedName := selected[progress.lookup]
			lookup := progress.lookup
			var resumeErr error
			progress, resumeErr = progress.resolveLookup(ctx, selectedName)
			if resumeErr != nil {
				return codemode.Step{}, fmt.Errorf("resolve Monty name %q: %w", lookup, resumeErr)
			}
		case frontierCall:
			step, err := checkpointCall(progress, tools, selected)
			if err != nil {
				return codemode.Step{}, err
			}
			return step, nil
		case frontierFuture:
			_ = abandon(progress)
			return codemode.Step{}, errors.New("monty produced an unsupported pending-future frontier")
		default:
			_ = abandon(progress)
			return codemode.Step{}, fmt.Errorf("monty produced unsupported progress kind %d", progress.kind)
		}
	}
}

func checkpointCall(snapshot *frontier, tools []string, selected map[string]struct{}) (codemode.Step, error) {
	fail := func(err error) (codemode.Step, error) {
		if abandonErr := abandon(snapshot); abandonErr != nil {
			err = errors.Join(err, abandonErr)
		}
		return codemode.Step{}, err
	}
	if snapshot.call.OS {
		return fail(fmt.Errorf("monty OS function %q is unavailable in codemode", snapshot.call.Name))
	}
	if snapshot.call.Method {
		return fail(fmt.Errorf("monty method callback %q is unavailable in codemode", snapshot.call.Name))
	}
	if _, ok := selected[snapshot.call.Name]; !ok {
		return fail(fmt.Errorf("monty requested unselected tool %q", snapshot.call.Name))
	}
	if snapshot.call.Positional != 0 {
		return fail(fmt.Errorf("monty tool %q requires keyword arguments", snapshot.call.Name))
	}
	if snapshot.dump == nil {
		return fail(errors.New("monty call cannot serialize a checkpoint"))
	}
	dump, err := snapshot.dump()
	if err != nil {
		return fail(fmt.Errorf("serialize Monty checkpoint: %w", err))
	}
	if err := abandon(snapshot); err != nil {
		return codemode.Step{}, err
	}
	encoded, err := json.Marshal(durableCheckpoint{
		Version: checkpointVersion, Snapshot: dump, Tools: slices.Clone(tools),
		CallID: snapshot.call.ID, CallName: snapshot.call.Name,
	})
	if err != nil {
		return codemode.Step{}, fmt.Errorf("encode Monty checkpoint: %w", err)
	}
	return codemode.Step{
		Checkpoint: codemode.Checkpoint(encoded),
		Calls: []codemode.Call{{
			ID: snapshot.call.ID, Name: snapshot.call.Name, Input: cloneObject(snapshot.call.Input),
		}},
	}, nil
}

func decodeCheckpoint(encoded []byte) (durableCheckpoint, error) {
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	var value durableCheckpoint
	if err := decoder.Decode(&value); err != nil {
		return durableCheckpoint{}, fmt.Errorf("decode Monty checkpoint: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		return durableCheckpoint{}, fmt.Errorf("decode Monty checkpoint: trailing data: %w", err)
	}
	if value.Version != checkpointVersion || len(value.Snapshot) == 0 || value.CallID == "" || value.CallName == "" {
		return durableCheckpoint{}, errors.New("invalid Monty checkpoint")
	}
	tools, err := normalizedToolNames(value.Tools)
	if err != nil {
		return durableCheckpoint{}, fmt.Errorf("invalid Monty checkpoint tools: %w", err)
	}
	value.Tools = tools
	found := false
	for _, name := range tools {
		found = found || name == value.CallName
	}
	if !found {
		return durableCheckpoint{}, errors.New("invalid Monty checkpoint: call is not a selected tool")
	}
	return value, nil
}

func toolNames(tools []codemode.Tool) ([]string, error) {
	names := make([]string, len(tools))
	for index, tool := range tools {
		names[index] = tool.Name
	}
	return normalizedToolNames(names)
}

func normalizedToolNames(names []string) ([]string, error) {
	if len(names) == 0 {
		return nil, errors.New("gomonty executor requires selected tools")
	}
	seen := make(map[string]struct{}, len(names))
	result := make([]string, len(names))
	for index, name := range names {
		if !validIdentifier(name) {
			return nil, fmt.Errorf("invalid gomonty tool name %q", name)
		}
		if _, ok := seen[name]; ok {
			return nil, fmt.Errorf("duplicate gomonty tool %q", name)
		}
		seen[name] = struct{}{}
		result[index] = name
	}
	return result, nil
}

func validIdentifier(value string) bool {
	if value == "" {
		return false
	}
	for index, character := range []byte(value) {
		letter := character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z'
		if !letter && character != '_' && (index == 0 || character < '0' || character > '9') {
			return false
		}
	}
	return true
}

func dictToObject(value monty.Dict) (map[string]any, error) {
	result := make(map[string]any, len(value))
	for _, pair := range value {
		key, ok := pair.Key.Raw().(string)
		if !ok || pair.Key.Kind() != monty.ValueKind("string") {
			return nil, errors.New("monty keyword argument has a non-string key")
		}
		if _, exists := result[key]; exists {
			return nil, fmt.Errorf("duplicate Monty keyword argument %q", key)
		}
		converted, err := toJSONValue(pair.Value)
		if err != nil {
			return nil, fmt.Errorf("keyword argument %q: %w", key, err)
		}
		result[key] = converted
	}
	return result, nil
}

func toJSONValue(value monty.Value) (any, error) {
	switch value.Kind() {
	case monty.ValueKind("none"):
		return nil, nil
	case monty.ValueKind("bool"), monty.ValueKind("int"), monty.ValueKind("float"), monty.ValueKind("string"):
		return value.Raw(), nil
	case monty.ValueKind("big_int"):
		integer := value.Raw().(*big.Int)
		return integer.String(), nil
	case monty.ValueKind("bytes"):
		bytesValue := value.Raw().([]byte)
		return base64.StdEncoding.EncodeToString(bytesValue), nil
	case monty.ValueKind("list"):
		return valuesToArray(value.Raw().([]monty.Value))
	case monty.ValueKind("tuple"):
		return valuesToArray([]monty.Value(value.Raw().(monty.Tuple)))
	case monty.ValueKind("set"):
		return valuesToArray([]monty.Value(value.Raw().(monty.Set)))
	case monty.ValueKind("frozen_set"):
		return valuesToArray([]monty.Value(value.Raw().(monty.FrozenSet)))
	case monty.ValueKind("dict"):
		return dictToObject(value.Raw().(monty.Dict))
	case monty.ValueKind("named_tuple"):
		named := value.Raw().(monty.NamedTuple)
		if len(named.FieldNames) != len(named.Values) {
			return nil, errors.New("invalid Monty named tuple")
		}
		result := make(map[string]any, len(named.Values))
		for index, field := range named.FieldNames {
			if field == "" {
				return nil, errors.New("monty named tuple has an empty field")
			}
			if _, exists := result[field]; exists {
				return nil, fmt.Errorf("monty named tuple has duplicate field %q", field)
			}
			converted, err := toJSONValue(named.Values[index])
			if err != nil {
				return nil, fmt.Errorf("named tuple field %q: %w", field, err)
			}
			result[field] = converted
		}
		return result, nil
	case monty.ValueKind("path"), monty.ValueKind("repr"):
		return fmt.Sprint(value.Raw()), nil
	case monty.ValueKind("cycle"):
		cycle := value.Raw().(monty.Cycle)
		return cycle.Placeholder, nil
	case monty.ValueKind("date"), monty.ValueKind("datetime"), monty.ValueKind("timedelta"), monty.ValueKind("timezone"):
		return normalizeJSON(value.Raw())
	default:
		return nil, fmt.Errorf("monty value kind %q is not JSON-shaped", value.Kind())
	}
}

func valuesToArray(values []monty.Value) ([]any, error) {
	result := make([]any, len(values))
	for index, value := range values {
		converted, err := toJSONValue(value)
		if err != nil {
			return nil, fmt.Errorf("item %d: %w", index, err)
		}
		result[index] = converted
	}
	return result, nil
}

func normalizeJSON(value any) (any, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var result any
	if err := json.Unmarshal(encoded, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func resultMessage(content any) string {
	if message, ok := content.(string); ok && message != "" {
		return message
	}
	encoded, err := json.Marshal(content)
	if err != nil {
		return "host tool failed"
	}
	return string(encoded)
}

func cloneObject(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	result := make(map[string]any, len(value))
	for key, item := range value {
		result[key] = cloneValue(item)
	}
	return result
}

func cloneValue(value any) any {
	switch current := value.(type) {
	case map[string]any:
		return cloneObject(current)
	case []any:
		result := make([]any, len(current))
		for index, item := range current {
			result[index] = cloneValue(item)
		}
		return result
	case []byte:
		return slices.Clone(current)
	default:
		return current
	}
}

func cloneLimits(value *monty.ResourceLimits) *monty.ResourceLimits {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func abandon(progress *frontier) error {
	if progress == nil {
		return nil
	}
	if progress.abandon == nil {
		return errors.New("monty progress cannot be abandoned")
	}
	return progress.abandon()
}

var _ codemode.Executor = (*Executor)(nil)
