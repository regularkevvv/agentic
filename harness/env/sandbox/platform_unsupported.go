//go:build !darwin && !linux

package sandbox

import (
	"context"
	"errors"
	"runtime"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

func probeBackend() (string, error) {
	return "", errors.New("strict process sandbox is unsupported on " + runtime.GOOS)
}

func execute(context.Context, execution) (harnessenv.CommandResult, error) {
	return harnessenv.CommandResult{}, errors.New("strict process sandbox is unavailable")
}
