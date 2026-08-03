package gomonty

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

func TestOptionalModuleUsesReleasedPublicBoundaries(t *testing.T) {
	t.Parallel()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate GoMonty module")
	}
	root := filepath.Dir(filename)
	moduleFile, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(moduleFile), "replace ") {
		t.Fatal("GoMonty adapter module must not contain replace directives")
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
			if strings.HasPrefix(imported, "github.com/regularkevvv/agentic/internal/") ||
				strings.HasPrefix(imported, "github.com/regularkevvv/agentic/harness/internal/") {
				t.Errorf("%s imports internal package %q", path, imported)
			}
			if strings.HasPrefix(imported, "github.com/regularkevvv/agentic/tui") {
				t.Errorf("%s imports TUI package %q", path, imported)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
