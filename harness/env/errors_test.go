package env

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"testing"
)

func TestErrorsPreserveCodes(t *testing.T) {
	t.Parallel()
	err := Wrap("read", "missing", fs.ErrNotExist)
	if !HasCode(err, CodeNotFound) || !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("wrapped error = %v", err)
	}
}

func TestErrorFormattingWrapMappingsAndResourceHelpers(t *testing.T) {
	t.Parallel()
	if got := (&Error{Op: "exec", Err: errors.New("failed")}).Error(); got != "exec: failed" {
		t.Fatalf("pathless error = %q", got)
	}
	if got := (&Error{Op: "read", Path: "file", Err: errors.New("failed")}).Error(); got != "read file: failed" {
		t.Fatalf("path error = %q", got)
	}
	if Wrap("noop", "", nil) != nil {
		t.Fatal("nil Wrap returned an error")
	}
	for code, source := range map[Code]error{
		CodeExists:     fs.ErrExist,
		CodePermission: fs.ErrPermission,
		CodeClosed:     os.ErrClosed,
		CodeIO:         errors.New("io"),
	} {
		if err := Wrap("op", "path", source); !HasCode(err, code) || !errors.Is(err, source) {
			t.Fatalf("mapped %v = %v", code, err)
		}
	}
	for _, source := range []error{context.Canceled, context.DeadlineExceeded} {
		if err := Wrap("op", "path", source); !errors.Is(err, source) {
			t.Fatalf("cancellation identity = %v", err)
		}
	}
	if HasCode(errors.New("plain"), CodeIO) {
		t.Fatal("plain error had an environment code")
	}
	display := CanonicalResource{Scheme: "file", ID: "opaque", Display: "shown"}
	if !display.Valid() || display.String() != "shown" {
		t.Fatalf("display resource = %#v", display)
	}
	opaque := CanonicalResource{Scheme: "memory", ID: "/file"}
	if opaque.String() != "memory:/file" || (CanonicalResource{}).Valid() {
		t.Fatalf("opaque resource = %#v", opaque)
	}
	called := false
	factory := FactoryFunc(func(_ context.Context, id string) (Lease, error) {
		called = id == "session"
		return nil, ErrFactoryClosed
	})
	if _, err := factory.Open(context.Background(), "session"); !errors.Is(err, ErrFactoryClosed) || !called {
		t.Fatalf("factory adapter = %v called=%t", err, called)
	}
}
