// Package codemodee2e contains the credential-free scenario shared by the
// runnable Harness Codemode example and its opt-in end-to-end test.
package codemodee2e

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	monty "github.com/regularkevvv/gomonty"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	"github.com/regularkevvv/agentic/harness/capability"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	"github.com/regularkevvv/agentic/harness/codemode"
	codemodegomonty "github.com/regularkevvv/agentic/harness/codemode/gomonty"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

const (
	hostCapabilityID = "host_tools"
	selectedToolName = "selected_tool"
	codeToolName     = "run_code"
	program          = "selected_tool(value=40)['value'] + 2"

	// ExpectedOutput is emitted only after the scripted model observes the
	// actual run_code result produced by Monty.
	ExpectedOutput = "GoMonty completed the full Harness Codemode path with result 42."
)

// Options controls the one explicit native-runtime acquisition performed
// before the Harness is constructed.
type Options struct {
	PrepareMode monty.PrepareMode
}

// Report exposes evidence from every boundary crossed by Run.
type Report struct {
	Runtime      monty.PreparedRuntime
	SessionID    string
	Cursor       uint64
	State        harness.SessionState
	Capabilities []string
	Output       string
	ModelCalls   int32
	HostCalls    int32
}

// ParsePrepareMode validates the two acquisition modes supported by GoMonty.
func ParsePrepareMode(value string) (monty.PrepareMode, error) {
	switch monty.PrepareMode(value) {
	case monty.PrepareDownload:
		return monty.PrepareDownload, nil
	case monty.PrepareBuild:
		return monty.PrepareBuild, nil
	default:
		return "", fmt.Errorf("unknown prepare mode %q; use %q or %q", value, monty.PrepareDownload, monty.PrepareBuild)
	}
}

// Run crosses the complete integration boundary without an external model:
// Harness session -> run_code -> GoMonty -> Monty worker -> selected Go tool ->
// restored Monty checkpoint -> run_code result -> final model response.
func Run(ctx context.Context, options Options) (report Report, err error) {
	mode := options.PrepareMode
	if mode == "" {
		mode = monty.PrepareDownload
	}
	if _, parseErr := ParsePrepareMode(string(mode)); parseErr != nil {
		return Report{}, parseErr
	}

	prepared, err := monty.Prepare(ctx, monty.PrepareOptions{Mode: mode})
	if err != nil {
		return Report{}, fmt.Errorf("prepare GoMonty runtime: %w", err)
	}
	report.Runtime = prepared

	model := &resultCheckingModel{}
	agent := agentic.NewAgent(
		"Use run_code to execute the requested computation and report its result.",
		model,
	)

	var hostCalls atomic.Int32
	var hostObservedRuntime atomic.Bool
	tool, handler, err := agentic.ToolWithContext(
		selectedToolName,
		"Return the supplied integer in an object.",
		func(toolCtx context.Context, input selectedToolInput) (map[string]int64, error) {
			hostCalls.Add(1)
			runtime, ok := harnessruntime.FromContext(toolCtx)
			call, callOK := agentic.CurrentToolCall(toolCtx)
			if !ok || runtime.SessionID == "" || runtime.Environment == nil ||
				!callOK || call.Name != selectedToolName || call.ID == "" {
				return nil, errors.New("selected tool did not receive Harness runtime context")
			}
			if _, resumed := agentic.CurrentToolResume(toolCtx); resumed {
				return nil, errors.New("selected tool inherited the outer codemode resume context")
			}
			hostObservedRuntime.Store(true)
			return map[string]int64{"value": input.Value}, nil
		},
	)
	if err != nil {
		return report, fmt.Errorf("construct selected tool: %w", err)
	}
	hostTools := capability.Func{
		Name: hostCapabilityID,
		Apply: func(registry *capability.Registry) error {
			return registry.AddToolset(agentic.NewToolset().Add(tool, handler))
		},
	}
	modeCapability := codemode.New(codemode.Config{
		Order:         capability.Ordering{After: []string{hostCapabilityID}},
		SelectedTools: []string{selectedToolName},
		Executor:      codemodegomonty.New(codemodegomonty.Config{}),
	})

	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	if err != nil {
		return report, fmt.Errorf("construct memory environment: %w", err)
	}
	processors, err := spill.NewFactory(artifactmemory.New(), spill.Config{})
	if err != nil {
		return report, fmt.Errorf("construct result processors: %w", err)
	}
	runtime, err := harness.New(
		agent,
		harness.WithRuntime(harness.RuntimeConfig{
			Sessions:              storememory.New(),
			Codec:                 jsoncodec.New(),
			Events:                inproc.NewFactory(),
			Environments:          environments,
			ResultProcessors:      processors,
			Clock:                 system.NewClock(),
			IDs:                   system.NewIDs(),
			ToolCancellationGrace: time.Second,
		}),
		harness.WithCapabilities(hostTools, modeCapability),
	).Build()
	if err != nil {
		return report, fmt.Errorf("build Harness: %w", err)
	}
	report.Capabilities = runtime.Capabilities()
	if !slices.Equal(report.Capabilities, []string{hostCapabilityID, "codemode"}) {
		return report, fmt.Errorf("unexpected capability order %v", report.Capabilities)
	}

	session, err := runtime.NewSession(ctx)
	if err != nil {
		return report, fmt.Errorf("create Harness session: %w", err)
	}
	report.SessionID = session.ID()
	defer func() {
		if closeErr := session.Close(context.WithoutCancel(ctx)); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close Harness session: %w", closeErr))
		}
	}()

	execution, err := session.Prompt(
		ctx,
		agentic.NewTextMessage(agentic.RoleUser, "Use code mode to call selected_tool with 40 and add 2."),
	)
	if err != nil {
		return report, fmt.Errorf("run Harness Codemode session: %w", err)
	}
	if execution == nil || execution.Status != agentic.ExecutionCompleted || execution.Result == nil {
		return report, fmt.Errorf("harness execution did not complete: %#v", execution)
	}
	report.Output = execution.Result.Output
	report.ModelCalls = model.calls.Load()
	report.HostCalls = hostCalls.Load()
	if report.Output != ExpectedOutput {
		return report, fmt.Errorf("final output = %q, want %q", report.Output, ExpectedOutput)
	}
	if report.ModelCalls != 2 || report.HostCalls != 1 || !hostObservedRuntime.Load() {
		return report, fmt.Errorf(
			"integration counts model=%d host=%d runtime_context=%t, want 2/1/true",
			report.ModelCalls,
			report.HostCalls,
			hostObservedRuntime.Load(),
		)
	}

	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		return report, fmt.Errorf("snapshot Harness session: %w", err)
	}
	report.Cursor = snapshot.Cursor
	report.State = snapshot.State
	if snapshot.Cursor == 0 || snapshot.State != harness.SessionIdle {
		return report, fmt.Errorf("unexpected durable snapshot cursor=%d state=%s", snapshot.Cursor, snapshot.State)
	}
	if err := validateTranscript(snapshot.Messages); err != nil {
		return report, err
	}
	return report, nil
}

type selectedToolInput struct {
	Value int64 `json:"value" description:"Integer to return"`
}

type resultCheckingModel struct {
	calls atomic.Int32
}

func (*resultCheckingModel) Name() string { return "e2e:codemode-result-checker" }

func (m *resultCheckingModel) Request(_ context.Context, request *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	call := m.calls.Add(1)
	switch call {
	case 1:
		if request == nil || len(request.Tools) != 1 || request.Tools[0].Function.Name != codeToolName {
			return nil, fmt.Errorf("model-visible tools did not contain only %q", codeToolName)
		}
		return &agentic.ChatResponse{
			Model: m.Name(),
			Message: agentic.NewToolUseMessage(agentic.ToolUse{
				ID:    "run-code-e2e-1",
				Name:  codeToolName,
				Input: map[string]any{"code": program},
			}),
			FinishReason:    agentic.FinishReasonToolCalls,
			RawFinishReason: string(agentic.FinishReasonToolCalls),
		}, nil
	case 2:
		if err := requireResult(request); err != nil {
			return nil, err
		}
		return &agentic.ChatResponse{
			Model:           m.Name(),
			Message:         agentic.NewTextMessage(agentic.RoleAssistant, ExpectedOutput),
			FinishReason:    agentic.FinishReasonStop,
			RawFinishReason: string(agentic.FinishReasonStop),
		}, nil
	default:
		return nil, fmt.Errorf("unexpected model request %d", call)
	}
}

func requireResult(request *agentic.ChatRequest) error {
	if request == nil {
		return errors.New("second model request is nil")
	}
	for messageIndex := len(request.Messages) - 1; messageIndex >= 0; messageIndex-- {
		for _, result := range request.Messages[messageIndex].GetToolResults() {
			if result.Name != codeToolName {
				continue
			}
			if result.IsError {
				return fmt.Errorf("run_code returned an error: %s", result.Content)
			}
			var payload struct {
				Output json.Number `json:"output"`
			}
			decoder := json.NewDecoder(strings.NewReader(result.Content))
			decoder.UseNumber()
			if err := decoder.Decode(&payload); err != nil {
				return fmt.Errorf("decode run_code result: %w", err)
			}
			if payload.Output.String() != "42" {
				return fmt.Errorf("run_code output = %s, want 42", payload.Output)
			}
			return nil
		}
	}
	return errors.New("second model request is missing the run_code result")
}

func validateTranscript(messages []agentic.Message) error {
	var calls, results int
	for index := range messages {
		for _, call := range messages[index].GetToolUses() {
			if call.ID == "run-code-e2e-1" && call.Name == codeToolName {
				calls++
			}
		}
		for _, result := range messages[index].GetToolResults() {
			if result.ToolUseID == "run-code-e2e-1" && result.Name == codeToolName && !result.IsError {
				results++
			}
		}
	}
	if calls != 1 || results != 1 {
		return fmt.Errorf("durable transcript has run_code calls/results %d/%d, want 1/1", calls, results)
	}
	return nil
}
