// Package artifact defines ports for complete tool-result storage and
// session-scoped result processing. Model-facing access is contributed
// separately through the gated artifacts capability.
package artifact

import (
	"context"
	"errors"
	"fmt"
	"strings"

	agentic "github.com/regularkevvv/agentic"
)

type Handle string

func (h Handle) String() string { return string(h) }

type Store interface {
	// Put is idempotent for one session and key. Reusing a key with different
	// bytes is an error rather than silently changing an existing handle.
	Put(context.Context, string, string, []byte) (Handle, error)
	Get(context.Context, string, Handle) ([]byte, error)
}

// ProcessorFactory creates the tool-result projection used by one session.
// Harness depends on this port instead of constructing a spill strategy.
type ProcessorFactory interface {
	Open(context.Context, string) (agentic.ToolResultProcessor, error)
}

type ProcessorFactoryFunc func(context.Context, string) (agentic.ToolResultProcessor, error)

func (f ProcessorFactoryFunc) Open(ctx context.Context, sessionID string) (agentic.ToolResultProcessor, error) {
	return f(ctx, sessionID)
}

var (
	ErrInvalidSessionID = errors.New("invalid artifact session ID")
	ErrInvalidHandle    = errors.New("invalid artifact handle")
	ErrArtifactNotFound = errors.New("artifact not found")
	ErrArtifactConflict = errors.New("artifact key already stores different data")
)

func ValidateSessionID(sessionID string) error {
	if sessionID == "" || len(sessionID) > 128 {
		return fmt.Errorf("%w: %q", ErrInvalidSessionID, sessionID)
	}
	for _, r := range sessionID {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '_' || r == '-' {
			continue
		}
		return fmt.Errorf("%w: %q", ErrInvalidSessionID, sessionID)
	}
	return nil
}

func ValidateHandle(handle Handle) error {
	value := string(handle)
	if !strings.HasPrefix(value, "art_") || len(value) != 68 {
		return fmt.Errorf("%w: %q", ErrInvalidHandle, value)
	}
	for _, r := range strings.TrimPrefix(value, "art_") {
		if (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') {
			continue
		}
		return fmt.Errorf("%w: %q", ErrInvalidHandle, value)
	}
	return nil
}
