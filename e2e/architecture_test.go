// Package e2e is the module root for the live provider tests and the runnable
// examples. It holds no code of its own; this file exists to assert the
// boundary that justifies the module.
package e2e

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// TestE2EModuleImportsOnlyPublicRootPackages holds this module to the same
// contract a downstream consumer has. Reaching into the root module's internal
// packages would let a live test pass against a shape no caller can construct,
// which is the failure these tests exist to catch.
//
// The imports are read with the parser rather than `go list` because the
// provider tests are behind //go:build e2e, and a build-configuration-aware
// walk would skip exactly the files most likely to break the rule.
func TestE2EModuleImportsOnlyPublicRootPackages(t *testing.T) {
	t.Parallel()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate e2e module")
	}
	root := filepath.Dir(filename)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() || filepath.Ext(path) != ".go" {
			return walkErr
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		ast.Inspect(file, func(node ast.Node) bool {
			spec, ok := node.(*ast.ImportSpec)
			if !ok {
				return true
			}
			value, unquoteErr := strconv.Unquote(spec.Path.Value)
			if unquoteErr == nil && strings.HasPrefix(value, "github.com/regularkevvv/agentic/internal/") {
				t.Errorf("%s imports root internal package %q", filepath.ToSlash(relative), value)
			}
			return false
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
