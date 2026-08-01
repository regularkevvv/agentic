//go:build !unix

package jsonl

import (
	"errors"
	"io/fs"
	"os"
)

var errFileLocked = errors.New("memory lock is held")

type fileLock struct {
	file *os.File
	path string
}

func acquireFileLock(path string) (*fileLock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0o600)
	if errors.Is(err, fs.ErrExist) {
		return nil, errFileLocked
	}
	if err != nil {
		return nil, err
	}
	return &fileLock{file: file, path: path}, nil
}

func (l *fileLock) Close() error {
	if l == nil || l.file == nil {
		return nil
	}
	closeErr := l.file.Close()
	removeErr := os.Remove(l.path)
	l.file = nil
	if closeErr != nil {
		return closeErr
	}
	return removeErr
}
