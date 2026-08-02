package onnx

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

// TestONNXModuleImportsOnlyPublicRootPackages holds this module to the same
// contract a downstream consumer has. Reaching into the root module's internal
// packages would let this encoder be built against a shape no caller can
// construct, and would quietly make the root's internals part of a nested
// module's compatibility surface.
//
// The imports are read with the parser rather than with `go list` so the walk
// sees every file regardless of build configuration.
func TestONNXModuleImportsOnlyPublicRootPackages(t *testing.T) {
	t.Parallel()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate onnx module")
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
