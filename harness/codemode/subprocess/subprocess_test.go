package subprocess

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/harness/codemode"
)

func TestHelperProcess(t *testing.T) {
	if os.Getenv("AGENTIC_CODEMODE_HELPER") != "1" {
		return
	}
	mode := "success"
	for index, value := range os.Args {
		if value == "--" && index+1 < len(os.Args) {
			mode = os.Args[index+1]
		}
	}
	input, _ := io.ReadAll(os.Stdin)
	switch mode {
	case "success":
		var request struct {
			Action string `json:"action"`
		}
		_ = json.Unmarshal(input, &request)
		_ = json.NewEncoder(os.Stdout).Encode(response{Version: 1, Step: codemode.Step{Done: true, Output: request.Action}})
	case "malformed":
		_, _ = fmt.Fprint(os.Stdout, "{")
	case "trailing":
		_, _ = fmt.Fprint(os.Stdout, `{"version":1,"step":{"done":true}} {}`)
	case "version":
		_, _ = fmt.Fprint(os.Stdout, `{"version":2,"step":{"done":true}}`)
	case "oversized":
		_, _ = fmt.Fprint(os.Stdout, strings.Repeat("x", 4096))
	case "exit":
		_, _ = fmt.Fprint(os.Stderr, "executor failed")
		os.Exit(3)
	case "exit-big":
		_, _ = fmt.Fprint(os.Stderr, strings.Repeat("e", 4096))
		os.Exit(3)
	case "exit-empty":
		os.Exit(3)
	case "block":
		time.Sleep(10 * time.Second)
	}
	os.Exit(0)
}

func helper(t *testing.T, mode string, outputLimit int) *Executor {
	t.Helper()
	t.Setenv("AGENTIC_CODEMODE_HELPER", "1")
	executor, err := New(Config{
		Executable:     os.Args[0],
		Args:           []string{"-test.run=TestHelperProcess", "--", mode},
		MaxOutputBytes: outputLimit,
		MaxStderrBytes: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	return executor
}

func TestStartAndResumeProtocol(t *testing.T) {
	executor := helper(t, "success", 1024)
	step, err := executor.Start(context.Background(), codemode.Request{Code: "program"})
	if err != nil || !step.Done || step.Output != "start" {
		t.Fatalf("start = %#v, %v", step, err)
	}
	step, err = executor.Resume(context.Background(), []byte("checkpoint"), []codemode.CallResult{{ID: "one", Name: "tool", Content: map[string]any{"ok": true}}})
	if err != nil || step.Output != "resume" {
		t.Fatalf("resume = %#v, %v", step, err)
	}
}

func TestProtocolFailuresAndLimits(t *testing.T) {
	for mode, target := range map[string]error{
		"malformed": ErrProtocol,
		"trailing":  ErrProtocol,
		"version":   ErrProtocol,
		"oversized": ErrOutputLimit,
	} {
		t.Run(mode, func(t *testing.T) {
			limit := 1024
			if mode == "oversized" {
				limit = 32
			}
			_, err := helper(t, mode, limit).Start(context.Background(), codemode.Request{Code: "program"})
			if !errors.Is(err, target) {
				t.Fatalf("error = %v", err)
			}
		})
	}
	_, err := helper(t, "exit", 1024).Start(context.Background(), codemode.Request{Code: "program"})
	if err == nil || !strings.Contains(err.Error(), "executor failed") {
		t.Fatalf("exit error = %v", err)
	}
}

func TestCancellationAndValidation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err := helper(t, "block", 1024).Start(ctx, codemode.Request{Code: "program"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("cancellation = %v", err)
	}
	if _, err := New(Config{}); err == nil {
		t.Fatal("missing executable succeeded")
	}
	if _, err := New(Config{Executable: os.Args[0], MaxOutputBytes: -1}); err == nil {
		t.Fatal("negative output bound succeeded")
	}
	if defaults, err := New(Config{Executable: os.Args[0]}); err != nil || defaults.maxOutput == 0 || defaults.maxStderr == 0 {
		t.Fatalf("default bounds = %#v, %v", defaults, err)
	}
	var executor *Executor
	if _, err := executor.Start(context.Background(), codemode.Request{}); err == nil {
		t.Fatal("nil executor succeeded")
	}
}

func TestEncodingAndExitDiagnosticFrontiers(t *testing.T) {
	executor := helper(t, "success", 1024)
	if _, err := executor.Start(context.Background(), codemode.Request{Tools: []codemode.Tool{{Name: "bad", Parameters: map[string]any{"bad": make(chan int)}}}}); err == nil || !strings.Contains(err.Error(), "encode") {
		t.Fatalf("request encoding = %v", err)
	}
	if _, err := helper(t, "exit-big", 1024).Start(context.Background(), codemode.Request{}); err == nil || !strings.Contains(err.Error(), "stderr truncated") {
		t.Fatalf("truncated stderr = %v", err)
	}
	if _, err := helper(t, "exit-empty", 1024).Start(context.Background(), codemode.Request{}); err == nil || strings.Contains(err.Error(), ": :") {
		t.Fatalf("empty stderr = %v", err)
	}
	if cloned, err := cloneResults(nil); err != nil || cloned != nil {
		t.Fatal("empty results clone was non-nil")
	}
	if _, err := executor.Resume(context.Background(), nil, []codemode.CallResult{{ID: "bad", Name: "bad", Content: make(chan int)}}); err == nil || !strings.Contains(err.Error(), "encode") {
		t.Fatalf("result encoding = %v", err)
	}
}

func TestLimitedBufferDropsBeyondBound(t *testing.T) {
	buffer := newLimitedBuffer(3)
	if written, err := buffer.Write([]byte("abcdef")); err != nil || written != 6 {
		t.Fatalf("write = %d, %v", written, err)
	}
	if string(buffer.Bytes()) != "abc" || buffer.String() != "abc" || !buffer.Exceeded() {
		t.Fatalf("buffer = %q exceeded=%v", buffer.String(), buffer.Exceeded())
	}
}
