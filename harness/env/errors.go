package env

import (
	"context"
	"errors"
	"io/fs"
	"os"
)

type Code uint8

const (
	CodeUnknown Code = iota
	CodeInvalid
	CodeEscaped
	CodeNotFound
	CodeExists
	CodeNotDirectory
	CodePermission
	CodeUnsupported
	CodeClosed
	CodeIO
)

type Error struct {
	Code Code
	Op   string
	Path string
	Err  error
}

func (e *Error) Error() string {
	if e.Path == "" {
		return e.Op + ": " + e.Err.Error()
	}
	return e.Op + " " + e.Path + ": " + e.Err.Error()
}

func (e *Error) Unwrap() error { return e.Err }

func HasCode(err error, code Code) bool {
	var target *Error
	return errors.As(err, &target) && target.Code == code
}

// Wrap maps an adapter error into the closed backend-independent error code
// set while preserving cancellation identity.
func Wrap(op, path string, err error) error {
	if err == nil {
		return nil
	}
	code := CodeIO
	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return err
	case errors.Is(err, fs.ErrNotExist):
		code = CodeNotFound
	case errors.Is(err, fs.ErrExist):
		code = CodeExists
	case errors.Is(err, fs.ErrPermission):
		code = CodePermission
	case errors.Is(err, os.ErrClosed):
		code = CodeClosed
	}
	return &Error{Code: code, Op: op, Path: path, Err: err}
}
