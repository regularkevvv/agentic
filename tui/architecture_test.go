package tui_test

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

func TestTUIModuleDependencyBoundary(t *testing.T) {
	t.Parallel()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate TUI module")
	}
	root := filepath.Dir(filename)
	moduleFile, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	contents := string(moduleFile)
	if strings.Contains(contents, "replace ") {
		t.Fatal("tui/go.mod must exercise released dependencies without replace directives")
	}
	if strings.Contains(contents, "gomonty") || strings.Contains(contents, "purego") {
		t.Fatal("TUI module graph contains the optional native-backed GoMonty adapter")
	}

	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() || filepath.Ext(path) != ".go" {
			return walkErr
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
			if strings.HasPrefix(imported, "github.com/regularkevvv/agentic/internal/") {
				t.Errorf("%s imports root internal package %q", path, imported)
			}
			if strings.Contains(imported, "/gomonty") {
				t.Errorf("%s imports optional GoMonty package %q", path, imported)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
