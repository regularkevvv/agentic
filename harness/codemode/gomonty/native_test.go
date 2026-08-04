package gomonty

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	monty "github.com/regularkevvv/gomonty"

	"github.com/regularkevvv/agentic/harness/codemode"
)

type stubNativeRunner struct {
	progress monty.Progress
	err      error
	closed   int
	options  monty.StartOptions
}

func (r *stubNativeRunner) Start(_ context.Context, options monty.StartOptions) (monty.Progress, error) {
	r.options = options
	return r.progress, r.err
}

func (r *stubNativeRunner) Close() error {
	r.closed++
	return nil
}

func TestMontyEngineCompileStartRestoreBoundaries(t *testing.T) {
	compileErr := errors.New("compile failed")
	engine := montyEngine{compile: func(string, monty.CompileOptions) (nativeRunner, error) {
		return nil, compileErr
	}}
	if _, err := engine.Start(context.Background(), "code", monty.CompileOptions{}, nil); !errors.Is(err, compileErr) {
		t.Fatalf("compile error = %v", err)
	}

	startErr := errors.New("start failed")
	runner := &stubNativeRunner{err: startErr}
	engine.compile = func(code string, options monty.CompileOptions) (nativeRunner, error) {
		if code != "code" || options.ScriptName != "test.py" {
			t.Fatalf("compile inputs = %q %#v", code, options)
		}
		return runner, nil
	}
	limits := &monty.ResourceLimits{MaxMemory: 7}
	if _, err := engine.Start(context.Background(), "code", monty.CompileOptions{ScriptName: "test.py"}, limits); !errors.Is(err, startErr) || runner.closed != 1 || runner.options.Limits != limits {
		t.Fatalf("start error=%v closed=%d options=%#v", err, runner.closed, runner.options)
	}

	runner = &stubNativeRunner{progress: &monty.Complete{Output: monty.Int(42)}}
	engine.compile = func(string, monty.CompileOptions) (nativeRunner, error) { return runner, nil }
	frontier, err := engine.Start(context.Background(), "code", monty.CompileOptions{}, nil)
	if err != nil || frontier.kind != frontierComplete || frontier.output != int64(42) || runner.closed != 1 {
		t.Fatalf("frontier=%#v error=%v closed=%d", frontier, err, runner.closed)
	}
	if err := frontier.abandon(); err != nil {
		t.Fatalf("abandon complete: %v", err)
	}

	restoreErr := errors.New("restore failed")
	engine.restore = func([]byte) (monty.Progress, error) { return nil, restoreErr }
	if _, err := engine.Restore([]byte("snapshot")); !errors.Is(err, restoreErr) {
		t.Fatalf("restore error = %v", err)
	}
	engine.restore = func(snapshot []byte) (monty.Progress, error) {
		if string(snapshot) != "snapshot" {
			t.Fatalf("snapshot = %q", snapshot)
		}
		return &monty.Complete{Output: monty.String("done")}, nil
	}
	frontier, err = engine.Restore([]byte("snapshot"))
	if err != nil || frontier.output != "done" {
		t.Fatalf("restored frontier=%#v error=%v", frontier, err)
	}

	defaults := newMontyEngine()
	if defaults.compile == nil || defaults.restore == nil {
		t.Fatal("default Monty engine is incomplete")
	}
	// The production factory must remain callable without preparing or
	// downloading a runtime. An unprepared host returns an error; a developer
	// cache may make the same probe compile successfully.
	compiled, compileErr := defaults.compile("42", monty.CompileOptions{ScriptName: "coverage.py"})
	if compileErr == nil {
		if err := compiled.Close(); err != nil {
			t.Fatalf("close production runner: %v", err)
		}
	}
}

func TestWrapMontyProgressVariantsWithoutNativeRuntime(t *testing.T) {
	lookup, err := wrapMontyProgress(&monty.NameLookupSnapshot{VariableName: "tool"})
	if err != nil || lookup.kind != frontierLookup || lookup.lookup != "tool" {
		t.Fatalf("lookup=%#v error=%v", lookup, err)
	}
	for _, selected := range []bool{false, true} {
		if _, err := lookup.resolveLookup(context.Background(), selected); err == nil {
			t.Fatalf("unbacked lookup selected=%t unexpectedly resumed", selected)
		}
	}
	if err := lookup.abandon(); err != nil {
		t.Fatalf("abandon lookup: %v", err)
	}

	call, err := wrapMontyProgress(&monty.Snapshot{
		FunctionName: "tool", CallID: 7,
		Args:   []monty.Value{monty.Int(1)},
		Kwargs: monty.Dict{{Key: monty.String("value"), Value: monty.Int(2)}},
	})
	if err != nil || call.kind != frontierCall || call.call.ID != "7" || call.call.Name != "tool" ||
		call.call.Positional != 1 || !reflect.DeepEqual(call.call.Input, map[string]any{"value": int64(2)}) {
		t.Fatalf("call=%#v error=%v", call, err)
	}
	if _, err := call.dump(); err == nil {
		t.Fatal("unbacked call unexpectedly dumped")
	}
	for _, result := range []codemode.CallResult{
		{Content: map[string]any{"ok": true}},
		{Content: "failed", IsError: true},
		{Content: make(chan int)},
	} {
		if _, err := call.resumeCall(context.Background(), result); err == nil {
			t.Fatalf("unbacked call result %#v unexpectedly resumed", result)
		}
	}
	if err := call.abandon(); err != nil {
		t.Fatalf("abandon call: %v", err)
	}

	osCall, err := wrapMontyProgress(&monty.Snapshot{FunctionName: "open", IsOSFunction: true, IsMethodCall: true})
	if err != nil || !osCall.call.OS || !osCall.call.Method {
		t.Fatalf("OS call=%#v error=%v", osCall, err)
	}
	if _, err := wrapMontyProgress(&monty.Snapshot{FunctionName: "tool", Kwargs: monty.Dict{{Key: monty.Int(1), Value: monty.Int(2)}}}); err == nil || !strings.Contains(err.Error(), "non-string") {
		t.Fatalf("invalid call error = %v", err)
	}
	future, err := wrapMontyProgress(&monty.FutureSnapshot{})
	if err != nil || future.kind != frontierFuture {
		t.Fatalf("future=%#v error=%v", future, err)
	}
	if err := future.abandon(); err != nil {
		t.Fatalf("abandon future: %v", err)
	}

	if _, err := wrapMontyProgress(&monty.Complete{Output: monty.FunctionValue(monty.Function{Name: "tool"})}); err == nil || !strings.Contains(err.Error(), "not JSON-shaped") {
		t.Fatalf("opaque complete error = %v", err)
	}
	if _, err := wrapMontyProgress(nil); err == nil {
		t.Fatal("nil Monty progress succeeded")
	}
}

func TestAbandonMontyTerminalAndUnbackedFrontiers(t *testing.T) {
	if err := abandonMonty(nil); err != nil {
		t.Fatal(err)
	}
	if err := abandonMonty(&monty.Complete{}); err != nil {
		t.Fatal(err)
	}
	for _, progress := range []monty.Progress{
		&monty.Snapshot{},
		&monty.NameLookupSnapshot{},
		&monty.FutureSnapshot{},
	} {
		if err := abandonMonty(progress); err != nil {
			t.Fatalf("abandon %T error = %v", progress, err)
		}
	}
}
