package gomonty

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	monty "github.com/regularkevvv/gomonty"

	"github.com/regularkevvv/agentic/harness/codemode"
)

type fakeEngine struct {
	start       *frontier
	startErr    error
	restore     *frontier
	restoreErr  error
	code        string
	compile     monty.CompileOptions
	limits      *monty.ResourceLimits
	restored    []byte
	startCalled int
}

func (e *fakeEngine) Start(
	_ context.Context,
	code string,
	compile monty.CompileOptions,
	limits *monty.ResourceLimits,
) (*frontier, error) {
	e.startCalled++
	e.code, e.compile, e.limits = code, compile, limits
	return e.start, e.startErr
}

func (e *fakeEngine) Restore(snapshot []byte) (*frontier, error) {
	e.restored = append([]byte(nil), snapshot...)
	return e.restore, e.restoreErr
}

func executorWithEngine(engine engine) *Executor {
	return &Executor{compile: monty.CompileOptions{ScriptName: "test.py"}, limits: &monty.ResourceLimits{MaxMemory: 9}, engine: engine}
}

func checkpoint(t *testing.T, mutate func(*durableCheckpoint)) codemode.Checkpoint {
	t.Helper()
	value := durableCheckpoint{
		Version: checkpointVersion, Snapshot: []byte("snapshot"), Tools: []string{"tool"},
		CallID: "7", CallName: "tool",
	}
	if mutate != nil {
		mutate(&value)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func TestNewDefensivelyCopiesConfigWithoutPreparingRuntime(t *testing.T) {
	inputs := []string{"value"}
	annotations := monty.AssertMessageAnnotations(64)
	limits := &monty.ResourceLimits{MaxMemory: 1024, MaxDuration: time.Second}
	executor := New(Config{CompileOptions: monty.CompileOptions{
		Inputs: inputs, AssertMessageAnnotations: &annotations,
	}, Limits: limits})

	inputs[0] = "changed"
	annotations = 8
	limits.MaxMemory = 1
	if executor.compile.ScriptName != "agentic-codemode.py" ||
		executor.compile.Inputs[0] != "value" ||
		*executor.compile.AssertMessageAnnotations != 64 ||
		executor.limits.MaxMemory != 1024 {
		t.Fatalf("executor config was not copied: %#v %#v", executor.compile, executor.limits)
	}
}

func TestValueConversionPreservesJSONShapes(t *testing.T) {
	value := monty.DictValue(monty.Dict{
		{Key: monty.String("items"), Value: monty.List(monty.Int(1), monty.Bool(true), monty.None())},
		{Key: monty.String("large"), Value: monty.BigInt(new(big.Int).Lsh(big.NewInt(1), 80))},
		{Key: monty.String("bytes"), Value: monty.Bytes([]byte("ok"))},
	})
	converted, err := toJSONValue(value)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{
		"items": []any{int64(1), true, nil},
		"large": "1208925819614629174706176",
		"bytes": "b2s=",
	}
	if !reflect.DeepEqual(converted, want) {
		t.Fatalf("converted = %#v, want %#v", converted, want)
	}
}

func TestValueConversionRejectsNonObjectDictAndOpaqueValues(t *testing.T) {
	_, err := toJSONValue(monty.DictValue(monty.Dict{{Key: monty.Int(1), Value: monty.String("value")}}))
	if err == nil || !strings.Contains(err.Error(), "non-string") {
		t.Fatalf("dict error = %v", err)
	}
	_, err = toJSONValue(monty.FunctionValue(monty.Function{Name: "tool"}))
	if err == nil || !strings.Contains(err.Error(), "not JSON-shaped") {
		t.Fatalf("function error = %v", err)
	}
}

func TestValueConversionCoversSupportedCompositeVariants(t *testing.T) {
	name := "UTC"
	offset := int32(0)
	values := []monty.Value{
		monty.None(),
		monty.Bool(true),
		monty.Int(1),
		monty.Float(1.5),
		monty.String("value"),
		monty.TupleValue(monty.Int(1), monty.String("two")),
		monty.SetValue(monty.Set{monty.Int(1)}),
		monty.FrozenSetValue(monty.FrozenSet{monty.Int(1)}),
		monty.NamedTupleValue(monty.NamedTuple{TypeName: "Pair", FieldNames: []string{"left", "right"}, Values: []monty.Value{monty.Int(1), monty.Int(2)}}),
		monty.PathValue(monty.Path("path/to/file")),
		monty.ReprValue("<opaque>"),
		monty.CycleRefValue(monty.Cycle{ID: 1, Placeholder: "<cycle>"}),
		monty.DateValue(monty.Date{Year: 2026, Month: 8, Day: 1}),
		monty.DateTimeValue(monty.DateTime{Year: 2026, Month: 8, Day: 1, OffsetSeconds: &offset, TimezoneName: &name}),
		monty.TimeDeltaValue(monty.TimeDelta{Days: 1, Seconds: 2}),
		monty.TimeZoneValue(monty.TimeZone{OffsetSeconds: 0, Name: &name}),
	}
	for _, value := range values {
		if _, err := toJSONValue(value); err != nil {
			t.Fatalf("convert %q: %v", value.Kind(), err)
		}
	}

	errorsToCheck := []monty.Value{
		monty.Ellipsis(),
		monty.ExceptionValue(monty.Exception{Type: "RuntimeError"}),
		monty.TypeValue(monty.Type{Name: "Widget"}),
		monty.BuiltinFunctionValue(monty.BuiltinFunction{Name: "len"}),
		monty.FileHandleValue(monty.FileHandle{Path: "file"}),
	}
	for _, value := range errorsToCheck {
		if _, err := toJSONValue(value); err == nil {
			t.Fatalf("unsupported %q succeeded", value.Kind())
		}
	}
}

func TestValueConversionRejectsMalformedCompositeFrontiers(t *testing.T) {
	invalid := []monty.Value{
		monty.DictValue(monty.Dict{
			{Key: monty.String("same"), Value: monty.Int(1)},
			{Key: monty.String("same"), Value: monty.Int(2)},
		}),
		monty.DictValue(monty.Dict{{Key: monty.String("bad"), Value: monty.FunctionValue(monty.Function{Name: "tool"})}}),
		monty.List(monty.FunctionValue(monty.Function{Name: "tool"})),
		monty.NamedTupleValue(monty.NamedTuple{FieldNames: []string{"one"}, Values: nil}),
		monty.NamedTupleValue(monty.NamedTuple{FieldNames: []string{""}, Values: []monty.Value{monty.Int(1)}}),
		monty.NamedTupleValue(monty.NamedTuple{FieldNames: []string{"same", "same"}, Values: []monty.Value{monty.Int(1), monty.Int(2)}}),
		monty.NamedTupleValue(monty.NamedTuple{FieldNames: []string{"bad"}, Values: []monty.Value{monty.FunctionValue(monty.Function{Name: "tool"})}}),
	}
	for index, value := range invalid {
		if _, err := toJSONValue(value); err == nil {
			t.Fatalf("invalid composite %d succeeded", index)
		}
	}
	if _, err := normalizeJSON(make(chan int)); err == nil {
		t.Fatal("unencodable normalized JSON succeeded")
	}
	if cloneObject(nil) != nil {
		t.Fatal("nil object clone became non-nil")
	}
}

func TestCheckpointValidationPrecedesNativeRuntimeAccess(t *testing.T) {
	encoded, err := json.Marshal(durableCheckpoint{
		Version: checkpointVersion, Snapshot: []byte("opaque"), Tools: []string{"tool"},
		CallID: "7", CallName: "tool",
	})
	if err != nil {
		t.Fatal(err)
	}
	executor := New(Config{})
	_, err = executor.Resume(context.Background(), encoded, []codemode.CallResult{{ID: "wrong", Name: "tool"}})
	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("identity error = %v", err)
	}

	_, err = executor.Resume(context.Background(), append(encoded, []byte("{}")...), nil)
	if err == nil || !strings.Contains(err.Error(), "trailing") {
		t.Fatalf("trailing checkpoint error = %v", err)
	}
}

func TestToolCatalogValidation(t *testing.T) {
	executor := New(Config{})
	for _, test := range []struct {
		name  string
		tools []codemode.Tool
		want  string
	}{
		{name: "missing", want: "requires selected tools"},
		{name: "empty", tools: []codemode.Tool{{Name: " "}}, want: "invalid"},
		{name: "identifier", tools: []codemode.Tool{{Name: "bad name"}}, want: "invalid"},
		{name: "duplicate", tools: []codemode.Tool{{Name: "tool"}, {Name: "tool"}}, want: "duplicate"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := executor.Start(context.Background(), codemode.Request{Code: "1", Tools: test.tools})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestToolIdentifierMatchesCodemodeContract(t *testing.T) {
	for _, value := range []string{"tool", "_tool", "Tool2"} {
		if !validIdentifier(value) {
			t.Fatalf("valid identifier %q rejected", value)
		}
	}
	for _, value := range []string{"", "2tool", "bad-name", "bad name"} {
		if validIdentifier(value) {
			t.Fatalf("invalid identifier %q accepted", value)
		}
	}
}

func TestExecutorStartDrivesLookupsAndDefensivelyCopiesLimits(t *testing.T) {
	var resolutions []string
	done := &frontier{kind: frontierComplete, output: map[string]any{"ok": true}, abandon: func() error { return nil }}
	unknown := &frontier{kind: frontierLookup, lookup: "ordinary_name", abandon: func() error { return nil }}
	unknown.resolveLookup = func(_ context.Context, selected bool) (*frontier, error) {
		resolutions = append(resolutions, fmt.Sprintf("ordinary_name=%t", selected))
		return done, nil
	}
	selected := &frontier{kind: frontierLookup, lookup: "tool", abandon: func() error { return nil }}
	selected.resolveLookup = func(_ context.Context, isSelected bool) (*frontier, error) {
		resolutions = append(resolutions, fmt.Sprintf("tool=%t", isSelected))
		return unknown, nil
	}
	backend := &fakeEngine{start: selected}
	executor := executorWithEngine(backend)
	step, err := executor.Start(context.Background(), codemode.Request{Code: "program", Tools: []codemode.Tool{{Name: "tool"}}})
	if err != nil {
		t.Fatal(err)
	}
	if !step.Done || !reflect.DeepEqual(step.Output, map[string]any{"ok": true}) ||
		!reflect.DeepEqual(resolutions, []string{"tool=true", "ordinary_name=false"}) {
		t.Fatalf("step=%#v resolutions=%v", step, resolutions)
	}
	if backend.code != "program" || backend.compile.ScriptName != "test.py" || backend.limits.MaxMemory != 9 {
		t.Fatalf("backend inputs = %#v %#v %q", backend.compile, backend.limits, backend.code)
	}
	backend.limits.MaxMemory = 1
	if executor.limits.MaxMemory != 9 {
		t.Fatal("backend mutated executor limits")
	}
}

func TestExecutorStartReturnsCheckpointedCallAndCopiesInput(t *testing.T) {
	abandoned := 0
	input := map[string]any{"nested": map[string]any{"value": int64(1)}, "items": []any{"one"}}
	backend := &fakeEngine{start: &frontier{
		kind: frontierCall,
		call: callFrontier{ID: "7", Name: "tool", Input: input},
		dump: func() ([]byte, error) { return []byte("durable"), nil },
		abandon: func() error {
			abandoned++
			return nil
		},
	}}
	step, err := executorWithEngine(backend).Start(context.Background(), codemode.Request{Code: "tool(value=1)", Tools: []codemode.Tool{{Name: "tool"}}})
	if err != nil {
		t.Fatal(err)
	}
	if step.Done || len(step.Calls) != 1 || abandoned != 1 {
		t.Fatalf("step=%#v abandoned=%d", step, abandoned)
	}
	state, err := decodeCheckpoint(step.Checkpoint)
	if err != nil || string(state.Snapshot) != "durable" || state.CallID != "7" || state.CallName != "tool" {
		t.Fatalf("checkpoint=%#v error=%v", state, err)
	}
	step.Calls[0].Input["nested"].(map[string]any)["value"] = int64(2)
	step.Calls[0].Input["items"].([]any)[0] = "changed"
	if input["nested"].(map[string]any)["value"] != int64(1) || input["items"].([]any)[0] != "one" {
		t.Fatal("returned call input aliases engine state")
	}
}

func TestCheckpointEncodingFailureIsReported(t *testing.T) {
	want := errors.New("encode failed")
	previous := marshalCheckpoint
	marshalCheckpoint = func(any) ([]byte, error) { return nil, want }
	t.Cleanup(func() { marshalCheckpoint = previous })
	_, err := checkpointCall(
		&frontier{
			kind:    frontierCall,
			call:    callFrontier{ID: "7", Name: "tool"},
			dump:    func() ([]byte, error) { return []byte("snapshot"), nil },
			abandon: func() error { return nil },
		},
		[]string{"tool"},
		map[string]struct{}{"tool": {}},
	)
	if !errors.Is(err, want) {
		t.Fatalf("checkpoint encode error = %v", err)
	}
}

func TestExecutorStartAndAdvanceFailures(t *testing.T) {
	var nilExecutor *Executor
	if _, err := nilExecutor.Start(context.Background(), codemode.Request{}); err == nil {
		t.Fatal("nil executor started")
	}
	backendErr := errors.New("backend failed")
	if _, err := executorWithEngine(&fakeEngine{startErr: backendErr}).Start(context.Background(), codemode.Request{Code: "x", Tools: []codemode.Tool{{Name: "tool"}}}); !errors.Is(err, backendErr) {
		t.Fatalf("backend error = %v", err)
	}

	abandonErr := errors.New("abandon failed")
	tests := []struct {
		name     string
		frontier *frontier
		want     string
	}{
		{name: "nil progress", want: "nil progress"},
		{name: "lookup without resume", frontier: &frontier{kind: frontierLookup, abandon: func() error { return nil }}, want: "cannot resume"},
		{name: "lookup resume error", frontier: &frontier{kind: frontierLookup, lookup: "name", resolveLookup: func(context.Context, bool) (*frontier, error) { return nil, backendErr }, abandon: func() error { return nil }}, want: "backend failed"},
		{name: "future", frontier: &frontier{kind: frontierFuture, abandon: func() error { return nil }}, want: "pending-future"},
		{name: "unknown", frontier: &frontier{kind: frontierKind(99), abandon: func() error { return nil }}, want: "unsupported progress kind"},
		{name: "os", frontier: failingCall(callFrontier{ID: "7", Name: "tool", OS: true}, nil), want: "OS function"},
		{name: "method", frontier: failingCall(callFrontier{ID: "7", Name: "tool", Method: true}, nil), want: "method callback"},
		{name: "unselected", frontier: failingCall(callFrontier{ID: "7", Name: "other"}, nil), want: "unselected"},
		{name: "positional", frontier: failingCall(callFrontier{ID: "7", Name: "tool", Positional: 1}, nil), want: "keyword arguments"},
		{name: "no dump", frontier: &frontier{kind: frontierCall, call: callFrontier{ID: "7", Name: "tool"}, abandon: func() error { return nil }}, want: "cannot serialize"},
		{name: "dump error", frontier: failingCall(callFrontier{ID: "7", Name: "tool"}, backendErr), want: "backend failed"},
		{name: "abandon after dump", frontier: &frontier{kind: frontierCall, call: callFrontier{ID: "7", Name: "tool"}, dump: func() ([]byte, error) { return []byte("snapshot"), nil }, abandon: func() error { return abandonErr }}, want: "abandon failed"},
		{name: "joined abandon", frontier: &frontier{kind: frontierCall, call: callFrontier{ID: "7", Name: "tool", OS: true}, abandon: func() error { return abandonErr }}, want: "abandon failed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := advance(context.Background(), test.frontier, []string{"tool"})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	abandoned := 0
	_, err := advance(canceled, &frontier{kind: frontierComplete, abandon: func() error { abandoned++; return nil }}, []string{"tool"})
	if !errors.Is(err, context.Canceled) || abandoned != 1 {
		t.Fatalf("canceled advance = %v, abandoned=%d", err, abandoned)
	}
}

func failingCall(call callFrontier, dumpErr error) *frontier {
	return &frontier{
		kind: frontierCall, call: call,
		dump: func() ([]byte, error) {
			if dumpErr != nil {
				return nil, dumpErr
			}
			return []byte("snapshot"), nil
		},
		abandon: func() error { return nil },
	}
}

func TestExecutorResumeRestoresAndDrivesResult(t *testing.T) {
	var got codemode.CallResult
	abandoned := 0
	backend := &fakeEngine{restore: &frontier{
		kind: frontierCall, call: callFrontier{ID: "7", Name: "tool"},
		resumeCall: func(_ context.Context, result codemode.CallResult) (*frontier, error) {
			got = result
			return &frontier{kind: frontierComplete, output: int64(42), abandon: func() error { return nil }}, nil
		},
		abandon: func() error { abandoned++; return nil },
	}}
	result := codemode.CallResult{ID: "7", Name: "tool", Content: map[string]any{"value": 42}, IsError: true}
	step, err := executorWithEngine(backend).Resume(context.Background(), checkpoint(t, nil), []codemode.CallResult{result})
	if err != nil {
		t.Fatal(err)
	}
	if !step.Done || step.Output != int64(42) || !reflect.DeepEqual(got, result) || string(backend.restored) != "snapshot" || abandoned != 0 {
		t.Fatalf("step=%#v result=%#v restored=%q abandoned=%d", step, got, backend.restored, abandoned)
	}
}

func TestExecutorResumeFailures(t *testing.T) {
	var nilExecutor *Executor
	if _, err := nilExecutor.Resume(context.Background(), nil, nil); err == nil {
		t.Fatal("nil executor resumed")
	}
	validResult := []codemode.CallResult{{ID: "7", Name: "tool"}}
	for _, count := range []int{0, 2} {
		results := make([]codemode.CallResult, count)
		if _, err := executorWithEngine(&fakeEngine{}).Resume(context.Background(), checkpoint(t, nil), results); err == nil || !strings.Contains(err.Error(), "exactly one") {
			t.Fatalf("count %d error = %v", count, err)
		}
	}

	backendErr := errors.New("restore failed")
	tests := []struct {
		name    string
		backend *fakeEngine
		want    string
	}{
		{name: "restore", backend: &fakeEngine{restoreErr: backendErr}, want: "restore failed"},
		{name: "nil", backend: &fakeEngine{}, want: "not a function call"},
		{name: "complete", backend: &fakeEngine{restore: &frontier{kind: frontierComplete, abandon: func() error { return nil }}}, want: "not a function call"},
		{name: "os", backend: restoredCall(callFrontier{ID: "7", Name: "tool", OS: true}, nil, nil), want: "differs"},
		{name: "method", backend: restoredCall(callFrontier{ID: "7", Name: "tool", Method: true}, nil, nil), want: "differs"},
		{name: "name", backend: restoredCall(callFrontier{ID: "7", Name: "other"}, nil, nil), want: "differs"},
		{name: "id", backend: restoredCall(callFrontier{ID: "8", Name: "tool"}, nil, nil), want: "differs"},
		{name: "no resume", backend: restoredCall(callFrontier{ID: "7", Name: "tool"}, nil, nil), want: "cannot resume"},
		{name: "resume error", backend: restoredCall(callFrontier{ID: "7", Name: "tool"}, func(context.Context, codemode.CallResult) (*frontier, error) { return nil, backendErr }, nil), want: "restore failed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := executorWithEngine(test.backend).Resume(context.Background(), checkpoint(t, nil), validResult)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func restoredCall(call callFrontier, resume func(context.Context, codemode.CallResult) (*frontier, error), abandonErr error) *fakeEngine {
	return &fakeEngine{restore: &frontier{
		kind: frontierCall, call: call, resumeCall: resume,
		abandon: func() error { return abandonErr },
	}}
}

func TestCheckpointDecoderRejectsInvalidFrontiers(t *testing.T) {
	invalid := []codemode.Checkpoint{
		nil,
		[]byte(`{"version":1,"unknown":true}`),
		checkpoint(t, func(value *durableCheckpoint) { value.Version = 2 }),
		checkpoint(t, func(value *durableCheckpoint) { value.Snapshot = nil }),
		checkpoint(t, func(value *durableCheckpoint) { value.CallID = "" }),
		checkpoint(t, func(value *durableCheckpoint) { value.CallName = "" }),
		checkpoint(t, func(value *durableCheckpoint) { value.Tools = nil }),
		checkpoint(t, func(value *durableCheckpoint) { value.Tools = []string{"tool", "tool"} }),
		checkpoint(t, func(value *durableCheckpoint) { value.Tools = []string{"other"} }),
	}
	for index, encoded := range invalid {
		if _, err := decodeCheckpoint(encoded); err == nil {
			t.Fatalf("invalid checkpoint %d succeeded", index)
		}
	}
}

func TestAbandonAndCloneHelpers(t *testing.T) {
	if err := abandon(nil); err != nil {
		t.Fatal(err)
	}
	if err := abandon(&frontier{}); err == nil {
		t.Fatal("frontier without abandon succeeded")
	}
	want := errors.New("closed")
	if err := abandon(&frontier{abandon: func() error { return want }}); !errors.Is(err, want) {
		t.Fatalf("abandon error = %v", err)
	}
	if cloneLimits(nil) != nil {
		t.Fatal("nil limits became non-nil")
	}
	original := map[string]any{"bytes": []byte("ok"), "items": []any{map[string]any{"value": int64(1)}}}
	cloned := cloneObject(original)
	cloned["bytes"].([]byte)[0] = 'x'
	cloned["items"].([]any)[0].(map[string]any)["value"] = int64(2)
	if string(original["bytes"].([]byte)) != "ok" || original["items"].([]any)[0].(map[string]any)["value"] != int64(1) {
		t.Fatal("clone aliases original")
	}
}

func TestExecutorCheckpointRoundTripWithPreparedRuntime(t *testing.T) {
	if os.Getenv("GOMONTY_INTEGRATION") != "1" {
		t.Skip("set GOMONTY_INTEGRATION=1 with an explicitly prepared runtime")
	}
	if _, err := monty.Prepared(); err != nil {
		t.Fatalf("verify explicitly prepared runtime: %v", err)
	}

	executor := New(Config{Limits: &monty.ResourceLimits{MaxDuration: 5 * time.Second}})
	step, err := executor.Start(context.Background(), codemode.Request{
		Code:  "selected_tool(value=40)['value'] + 2",
		Tools: []codemode.Tool{{Name: "selected_tool"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if step.Done || len(step.Checkpoint) == 0 || len(step.Calls) != 1 {
		t.Fatalf("start step = %#v", step)
	}
	call := step.Calls[0]
	if call.Name != "selected_tool" || !reflect.DeepEqual(call.Input, map[string]any{"value": int64(40)}) {
		t.Fatalf("call = %#v", call)
	}

	// A fresh adapter proves that only the durable snapshot, not process-local
	// executor state, is required to resume.
	resumed, err := New(Config{}).Resume(context.Background(), step.Checkpoint, []codemode.CallResult{{
		ID: call.ID, Name: call.Name, Content: map[string]any{"value": int64(40)},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !resumed.Done || resumed.Output != int64(42) {
		t.Fatalf("resumed step = %#v", resumed)
	}
}

func TestResultMessageIsDeterministicEnoughForExceptions(t *testing.T) {
	if got := resultMessage("failed"); got != "failed" {
		t.Fatalf("message = %q", got)
	}
	if got := resultMessage(make(chan int)); got != "host tool failed" {
		t.Fatalf("fallback = %q", got)
	}
	if err := errors.New(resultMessage(map[string]any{"failed": true})); err.Error() != `{"failed":true}` {
		t.Fatalf("object message = %q", err)
	}
}
