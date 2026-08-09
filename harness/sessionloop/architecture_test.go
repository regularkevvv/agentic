package sessionloop_test

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func moduleRoot(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate the sessionloop module")
	}
	return filepath.Dir(filename)
}

// TestModuleDeclaresNoRequirements freezes the module's core promise: the
// sessionloop protocol has zero dependencies, so importing it never places
// Agentic, Harness, the TUI, a provider SDK, or any third-party module in a
// consumer's module graph. Per the plan (docs/design/harness-sessionloop-plan.md
// section 5.3), adding a require or replace directive to this go.mod is an
// architectural revision that must update the plan first; this test fails
// until that revision is explicit.
func TestModuleDeclaresNoRequirements(t *testing.T) {
	t.Parallel()
	contents, err := os.ReadFile(filepath.Join(moduleRoot(t), "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(contents), "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "" || strings.HasPrefix(trimmed, "//"):
		case strings.HasPrefix(trimmed, "module "):
		case strings.HasPrefix(trimmed, "go "):
		case strings.HasPrefix(trimmed, "toolchain "):
		default:
			t.Errorf("go.mod line %q is neither the module path, the go directive, nor a comment; "+
				"harness/sessionloop must stay a zero-require, no-replace module, and adding a "+
				"requirement demands an explicit architectural revision of "+
				"docs/design/harness-sessionloop-plan.md section 5.3 first", trimmed)
		}
	}
}

// TestModuleImportsOnlyTheStandardLibraryAndItself walks every Go file in
// the module (testdata excluded: it is release-workflow input, not module
// code) and rejects any import outside the standard library or the module's
// own packages. Source-level neutrality plus the empty go.mod together keep
// the published dependency promise.
func TestModuleImportsOnlyTheStandardLibraryAndItself(t *testing.T) {
	t.Parallel()
	root := moduleRoot(t)
	const modulePath = "github.com/regularkevvv/agentic/harness/sessionloop"
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if entry.Name() == "testdata" {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" {
			return nil
		}
		file, parseErr := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if parseErr != nil {
			return parseErr
		}
		for _, spec := range file.Imports {
			imported, unquoteErr := strconv.Unquote(spec.Path.Value)
			if unquoteErr != nil {
				return unquoteErr
			}
			if imported == modulePath || strings.HasPrefix(imported, modulePath+"/") {
				continue
			}
			first := imported
			if index := strings.Index(imported, "/"); index >= 0 {
				first = imported[:index]
			}
			if strings.Contains(first, ".") {
				t.Errorf("%s imports %q; harness/sessionloop allows only standard-library imports "+
					"and its own packages so consumers never inherit a dependency", path, imported)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
