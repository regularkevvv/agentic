package env

import (
	"errors"
	"io/fs"
	"testing"
)

func TestErrorsPreserveCodes(t *testing.T) {
	t.Parallel()
	err := Wrap("read", "missing", fs.ErrNotExist)
	if !HasCode(err, CodeNotFound) || !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("wrapped error = %v", err)
	}
}
