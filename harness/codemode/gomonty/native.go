package gomonty

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	monty "github.com/regularkevvv/gomonty"

	"github.com/regularkevvv/agentic/harness/codemode"
)

type nativeRunner interface {
	Start(context.Context, monty.StartOptions) (monty.Progress, error)
	Close() error
}

type montyEngine struct {
	compile func(string, monty.CompileOptions) (nativeRunner, error)
	restore func([]byte) (monty.Progress, error)
}

func newMontyEngine() montyEngine {
	return montyEngine{
		compile: func(code string, options monty.CompileOptions) (nativeRunner, error) {
			return monty.New(code, options)
		},
		restore: monty.LoadSnapshot,
	}
}

func (e montyEngine) Start(
	ctx context.Context,
	code string,
	compile monty.CompileOptions,
	limits *monty.ResourceLimits,
) (*frontier, error) {
	runner, err := e.compile(code, compile)
	if err != nil {
		return nil, fmt.Errorf("compile Monty program: %w", err)
	}
	defer func() { _ = runner.Close() }()
	progress, err := runner.Start(ctx, monty.StartOptions{Limits: limits})
	if err != nil {
		return nil, fmt.Errorf("start Monty program: %w", err)
	}
	return wrapMontyProgress(progress)
}

func (e montyEngine) Restore(snapshot []byte) (*frontier, error) {
	progress, err := e.restore(snapshot)
	if err != nil {
		return nil, err
	}
	return wrapMontyProgress(progress)
}

func wrapMontyProgress(progress monty.Progress) (*frontier, error) {
	switch current := progress.(type) {
	case *monty.Complete:
		output, err := toJSONValue(current.Output)
		if err != nil {
			return nil, fmt.Errorf("convert Monty output: %w", err)
		}
		return &frontier{kind: frontierComplete, output: output, abandon: func() error { return nil }}, nil
	case *monty.NameLookupSnapshot:
		return &frontier{
			kind:   frontierLookup,
			lookup: current.VariableName,
			resolveLookup: func(ctx context.Context, selected bool) (*frontier, error) {
				var next monty.Progress
				var err error
				if selected {
					next, err = current.ResumeValue(ctx, monty.FunctionValue(monty.Function{Name: current.VariableName}))
				} else {
					next, err = current.ResumeUndefined(ctx)
				}
				if err != nil {
					return nil, err
				}
				return wrapMontyProgress(next)
			},
			abandon: func() error { return abandonMonty(current) },
		}, nil
	case *monty.Snapshot:
		input, err := dictToObject(current.Kwargs)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("convert Monty arguments for %q: %w", current.FunctionName, err), abandonMonty(current))
		}
		return &frontier{
			kind: frontierCall,
			call: callFrontier{
				ID: strconv.FormatUint(uint64(current.CallID), 10), Name: current.FunctionName,
				Input: input, Positional: len(current.Args), OS: current.IsOSFunction, Method: current.IsMethodCall,
			},
			dump: current.Dump,
			resumeCall: func(ctx context.Context, result codemode.CallResult) (*frontier, error) {
				var next monty.Progress
				var resumeErr error
				if result.IsError {
					message := resultMessage(result.Content)
					next, resumeErr = current.ResumeException(ctx, monty.Exception{Type: "RuntimeError", Arg: &message})
				} else {
					value, err := monty.ValueOf(result.Content)
					if err != nil {
						return nil, errors.Join(fmt.Errorf("convert host result for Monty: %w", err), abandonMonty(current))
					}
					next, resumeErr = current.ResumeReturn(ctx, value)
				}
				if resumeErr != nil {
					return nil, resumeErr
				}
				return wrapMontyProgress(next)
			},
			abandon: func() error { return abandonMonty(current) },
		}, nil
	case *monty.FutureSnapshot:
		return &frontier{kind: frontierFuture, abandon: func() error { return abandonMonty(current) }}, nil
	case nil:
		return nil, errors.New("monty produced nil progress")
	default:
		return nil, fmt.Errorf("monty produced unsupported progress %T", progress)
	}
}

func abandonMonty(progress monty.Progress) error {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	var err error
	switch current := progress.(type) {
	case *monty.Snapshot:
		_, err = current.ResumeReturn(canceled, monty.None())
	case *monty.NameLookupSnapshot:
		_, err = current.ResumeUndefined(canceled)
	case *monty.FutureSnapshot:
		_, err = current.ResumeResults(canceled, nil)
	case *monty.Complete, nil:
		return nil
	default:
		return fmt.Errorf("cannot abandon Monty progress %T", progress)
	}
	if errors.Is(err, context.Canceled) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("abandon Monty progress: %w", err)
	}
	return nil
}

var _ engine = montyEngine{}
