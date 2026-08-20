//go:build darwin

package sandbox

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSeatbeltProfileHasNoUnfilteredHostRead(t *testing.T) {
	if strings.Contains(seatbeltProfile, "\n(allow file-read*)\n") {
		t.Fatal("seatbelt profile grants unfiltered host reads")
	}
	for _, required := range []string{"WORKSPACE", "PRIVATE_TMP", "(deny default)", "/usr", "/System"} {
		if !strings.Contains(seatbeltProfile, required) {
			t.Fatalf("seatbelt profile lacks %q", required)
		}
	}
}

func TestValidateBackendExecutable(t *testing.T) {
	if err := validateBackendExecutable(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("missing backend executable was accepted")
	}
	root := t.TempDir()
	if err := validateBackendExecutable(root); err == nil {
		t.Fatal("backend directory was accepted as executable")
	}
	file := filepath.Join(root, "plain-file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := validateBackendExecutable(file); err == nil {
		t.Fatal("non-executable backend file was accepted")
	}
	if err := validateBackendExecutable("/usr/bin/sandbox-exec"); err != nil {
		t.Fatalf("system Seatbelt backend = %v", err)
	}
}
