// Package subprocess adapts the codemode Executor port to one fixed binary.
// A child process provides crash isolation only; it is not an OS sandbox.
package subprocess

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"

	"github.com/regularkevvv/agentic/harness/codemode"
)

const protocolVersion = 1

var (
	ErrOutputLimit = errors.New("codemode subprocess output limit exceeded")
	ErrProtocol    = errors.New("invalid codemode subprocess protocol")
)

type Config struct {
	Executable     string
	Args           []string
	MaxOutputBytes int
	MaxStderrBytes int
}

type Executor struct {
	executable string
	args       []string
	maxOutput  int
	maxStderr  int
}

func New(config Config) (*Executor, error) {
	if strings.TrimSpace(config.Executable) == "" {
		return nil, errors.New("codemode subprocess executable is required")
	}
	if config.MaxOutputBytes == 0 {
		config.MaxOutputBytes = 2 << 20
	}
	if config.MaxStderrBytes == 0 {
		config.MaxStderrBytes = 64 << 10
	}
	if config.MaxOutputBytes <= 0 || config.MaxStderrBytes <= 0 {
		return nil, errors.New("codemode subprocess output limits must be positive")
	}
	return &Executor{
		executable: config.Executable,
		args:       append([]string(nil), config.Args...),
		maxOutput:  config.MaxOutputBytes,
		maxStderr:  config.MaxStderrBytes,
	}, nil
}

type request struct {
	Version    int                   `json:"version"`
	Action     string                `json:"action"`
	Start      *codemode.Request     `json:"start,omitempty"`
	Checkpoint codemode.Checkpoint   `json:"checkpoint,omitempty"`
	Results    []codemode.CallResult `json:"results,omitempty"`
}

type response struct {
	Version int           `json:"version"`
	Step    codemode.Step `json:"step"`
}

func (e *Executor) Start(ctx context.Context, start codemode.Request) (codemode.Step, error) {
	return e.invoke(ctx, request{Version: protocolVersion, Action: "start", Start: &start})
}

func (e *Executor) Resume(
	ctx context.Context,
	checkpoint codemode.Checkpoint,
	results []codemode.CallResult,
) (codemode.Step, error) {
	cloned, err := cloneResults(results)
	if err != nil {
		return codemode.Step{}, err
	}
	return e.invoke(ctx, request{
		Version: protocolVersion, Action: "resume",
		Checkpoint: append(codemode.Checkpoint(nil), checkpoint...),
		Results:    cloned,
	})
}

func (e *Executor) invoke(ctx context.Context, value request) (codemode.Step, error) {
	if e == nil {
		return codemode.Step{}, errors.New("codemode subprocess executor is nil")
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return codemode.Step{}, fmt.Errorf("encode codemode subprocess request: %w", err)
	}
	command := exec.CommandContext(ctx, e.executable, e.args...)
	command.Stdin = bytes.NewReader(encoded)
	stdout := newLimitedBuffer(e.maxOutput)
	stderr := newLimitedBuffer(e.maxStderr)
	command.Stdout = stdout
	command.Stderr = stderr
	runErr := command.Run()
	if ctx.Err() != nil {
		return codemode.Step{}, ctx.Err()
	}
	if stdout.Exceeded() {
		return codemode.Step{}, ErrOutputLimit
	}
	if runErr != nil {
		message := strings.TrimSpace(stderr.String())
		if stderr.Exceeded() {
			message += " (stderr truncated)"
		}
		if message == "" {
			return codemode.Step{}, fmt.Errorf("codemode subprocess: %w", runErr)
		}
		return codemode.Step{}, fmt.Errorf("codemode subprocess: %w: %s", runErr, message)
	}
	decoder := json.NewDecoder(bytes.NewReader(stdout.Bytes()))
	decoder.DisallowUnknownFields()
	var decoded response
	if err := decoder.Decode(&decoded); err != nil {
		return codemode.Step{}, fmt.Errorf("%w: decode response: %v", ErrProtocol, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		return codemode.Step{}, fmt.Errorf("%w: trailing response: %v", ErrProtocol, err)
	}
	if decoded.Version != protocolVersion {
		return codemode.Step{}, fmt.Errorf("%w: got version %d", ErrProtocol, decoded.Version)
	}
	return decoded.Step, nil
}

type limitedBuffer struct {
	mu       sync.Mutex
	buffer   bytes.Buffer
	maximum  int
	exceeded bool
}

func newLimitedBuffer(maximum int) *limitedBuffer { return &limitedBuffer{maximum: maximum} }

func (b *limitedBuffer) Write(value []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	remaining := b.maximum - b.buffer.Len()
	if remaining > 0 {
		portion := value
		if len(portion) > remaining {
			portion = portion[:remaining]
		}
		_, _ = b.buffer.Write(portion)
	}
	if len(value) > remaining {
		b.exceeded = true
	}
	return len(value), nil
}

func (b *limitedBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.buffer.Bytes()...)
}

func (b *limitedBuffer) String() string { return string(b.Bytes()) }

func (b *limitedBuffer) Exceeded() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.exceeded
}

func cloneResults(results []codemode.CallResult) ([]codemode.CallResult, error) {
	if len(results) == 0 {
		return nil, nil
	}
	encoded, err := json.Marshal(results)
	if err != nil {
		return nil, fmt.Errorf("encode codemode subprocess results: %w", err)
	}
	var cloned []codemode.CallResult
	if err := json.Unmarshal(encoded, &cloned); err != nil {
		return nil, fmt.Errorf("clone codemode subprocess results: %w", err)
	}
	return cloned, nil
}
