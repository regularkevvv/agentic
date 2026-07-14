// Package envutil provides shared helpers for example programs.
package envutil

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"
)

// LoadDotEnv loads key=value pairs from a .env file into the environment.
// Missing files are silently ignored. Existing env vars are not overwritten.
func LoadDotEnv() error {
	file, err := os.Open(".env")
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

// PromptFromArgs returns command-line arguments joined as a string,
// or the fallback if no arguments were provided.
func PromptFromArgs(fallback string) string {
	if len(os.Args) > 1 {
		return strings.Join(os.Args[1:], " ")
	}
	return fallback
}
