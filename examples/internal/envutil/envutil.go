// Package envutil provides shared helpers for example programs.
package envutil

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// LoadDotEnv loads key=value pairs from a .env file into the environment.
// Missing files are silently ignored. Existing env vars are not overwritten.
//
// The search walks up from the working directory to the filesystem root and
// uses the first .env it finds, so one file at the repository root serves
// every example whether it is run as `go run ./examples/basic` from the root
// or as `go run .` from inside the example's own directory. Looking only in
// the working directory made the second form fail with a confusing
// missing-credential error from the provider.
func LoadDotEnv() error {
	path, ok := findDotEnv()
	if !ok {
		return nil
	}

	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer func() { _ = file.Close() }()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		key, value, found := strings.Cut(line, "=")
		if !found {
			return fmt.Errorf("invalid .env entry %q", line)
		}

		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			return err
		}
	}

	return scanner.Err()
}

// findDotEnv returns the nearest .env at or above the working directory.
func findDotEnv() (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for {
		candidate := filepath.Join(dir, ".env")
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

// PromptFromArgs returns command-line arguments joined as a string,
// or the fallback if no arguments were provided.
func PromptFromArgs(fallback string) string {
	if len(os.Args) > 1 {
		return strings.Join(os.Args[1:], " ")
	}
	return fallback
}
