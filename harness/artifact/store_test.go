package artifact

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentic "github.com/regularkevvv/agentic"
)

func TestHandleValidation(t *testing.T) {
	t.Parallel()
	if err := ValidateHandle("art_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateHandle("art_not-opaque"); err == nil {
		t.Fatal("invalid handle succeeded")
	}
}

func TestSessionHandleAndFactoryValidation(t *testing.T) {
	t.Parallel()
	if Handle("opaque").String() != "opaque" {
		t.Fatal("handle String changed")
	}
	for _, id := range []string{"", "bad/session", strings.Repeat("a", 129)} {
		if err := ValidateSessionID(id); !errors.Is(err, ErrInvalidSessionID) {
			t.Fatalf("session %q = %v", id, err)
		}
	}
	if err := ValidateSessionID("Valid_session-1"); err != nil {
		t.Fatal(err)
	}
	upper := Handle("art_" + strings.Repeat("A", 64))
	if err := ValidateHandle(upper); !errors.Is(err, ErrInvalidHandle) {
		t.Fatalf("uppercase handle = %v", err)
	}
	called := false
	wantErr := errors.New("open")
	factory := ProcessorFactoryFunc(func(_ context.Context, id string) (agentic.ToolResultProcessor, error) {
		called = id == "session"
		return nil, wantErr
	})
	if _, err := factory.Open(context.Background(), "session"); !errors.Is(err, wantErr) || !called {
		t.Fatalf("factory adapter = %v called=%t", err, called)
	}
}
