package agentic_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// capabilityInterfaces are the contracts a provider package can satisfy. A
// provider that satisfies none of them is not a provider.
var capabilityInterfaces = map[string]bool{
	"Model":                 true,
	"StreamModel":           true,
	"Embedder":              true,
	"RepresentationEncoder": true,
	"Reranker":              true,
}

// TestEveryProviderDeclaresItsCapabilities enforces the one thing the directory
// tree cannot say.
//
// provider/ is flat and most of its packages hold several roles at once —
// bedrock is a model and an embedder, cohere an embedder and a reranker — so no
// arrangement of directories can encode capability, and grouping by role would
// have to put those packages in two places. The compile-time assertion is
// therefore the only machine-readable answer to "what is this provider for",
// and this test is what stops it being optional. A new provider that forgets it
// fails here rather than being discovered by reading its source.
func TestEveryProviderDeclaresItsCapabilities(t *testing.T) {
	t.Parallel()

	_, filename, _, _ := runtime.Caller(0)
	providerRoot := filepath.Join(filepath.Dir(filename), "provider")

	entries, err := os.ReadDir(providerRoot)
	if err != nil {
		t.Fatalf("read provider directory: %v", err)
	}

	checked := 0
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(providerRoot, entry.Name())

		// A package with its own go.mod is a separate module and is compiled,
		// tested, and asserted by itself. Walking into it from here would test
		// source this module never builds.
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			continue
		}
		// provider/test holds the test doubles and the conformance suite rather
		// than a provider, and is exercised by the packages that consume it.
		if entry.Name() == "test" {
			continue
		}
		// A directory that groups providers rather than being one — provider/local
		// holds only the nested onnx module — has no Go files of its own and
		// nothing to assert.
		if !hasGoFiles(dir) {
			continue
		}

		declared, err := declaredCapabilities(dir)
		if err != nil {
			t.Fatalf("%s: %v", entry.Name(), err)
		}
		if len(declared) == 0 {
			t.Errorf("provider/%s declares no capability: add a compile-time "+
				"assertion such as `var _ retrieval.Embedder = (*Embedder)(nil)` so "+
				"what this package implements is checkable rather than folklore",
				entry.Name())
			continue
		}
		checked++
	}

	// A refactor that moved or renamed provider/ would otherwise leave this
	// test passing over nothing.
	if checked < 10 {
		t.Fatalf("checked only %d provider packages, expected the full set", checked)
	}
}

// hasGoFiles reports whether dir is itself a package rather than a directory
// that only groups others.
func hasGoFiles(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() && filepath.Ext(name) == ".go" && !strings.HasSuffix(name, "_test.go") {
			return true
		}
	}
	return false
}

// declaredCapabilities returns the capability interfaces a package asserts,
// reading `var _ core.X = ...` declarations from its non-test files. Nested
// modules and test files are excluded by the caller and by the walk.
func declaredCapabilities(dir string) (map[string]bool, error) {
	found := make(map[string]bool)
	err := filepath.WalkDir(dir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path == dir {
				return nil
			}
			if _, err := os.Stat(filepath.Join(path, "go.mod")); err == nil {
				return fs.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
			return nil
		}

		file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return err
		}
		for _, decl := range file.Decls {
			genDecl, ok := decl.(*ast.GenDecl)
			if !ok || genDecl.Tok != token.VAR {
				continue
			}
			for _, spec := range genDecl.Specs {
				value, ok := spec.(*ast.ValueSpec)
				if !ok || value.Type == nil {
					continue
				}
				// Only the blank identifier: `var _ retrieval.Embedder = ...` is an
				// assertion, whereas a named variable of that type is a field.
				if len(value.Names) != 1 || value.Names[0].Name != "_" {
					continue
				}
				selector, ok := value.Type.(*ast.SelectorExpr)
				if !ok {
					continue
				}
				if capabilityInterfaces[selector.Sel.Name] {
					found[selector.Sel.Name] = true
				}
			}
		}
		return nil
	})
	return found, err
}
