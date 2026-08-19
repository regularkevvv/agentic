//go:build linux && !amd64 && !arm64

package sandbox

import (
	"context"
	"errors"
	"runtime"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

func probeBackend() (string, error) {
	return "", errors.New("strict Linux sandbox is unsupported on " + runtime.GOARCH)
}

func execute(context.Context, execution) (harnessenv.CommandResult, error) {
	return harnessenv.CommandResult{}, errors.New("strict Linux sandbox is unavailable")
}
