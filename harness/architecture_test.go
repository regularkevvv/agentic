package harness

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

func TestCorePackagesDoNotImportAdapters(t *testing.T) {
	t.Parallel()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate harness module")
	}
	root := filepath.Dir(filename)
	adapterPrefixes := []string{
		"github.com/regularkevvv/agentic/harness/artifact/file",
		"github.com/regularkevvv/agentic/harness/artifact/memory",
		"github.com/regularkevvv/agentic/harness/artifact/spill",
		"github.com/regularkevvv/agentic/harness/codec/json",
		"github.com/regularkevvv/agentic/harness/codemode/subprocess",
		"github.com/regularkevvv/agentic/harness/env/local",
		"github.com/regularkevvv/agentic/harness/env/memory",
		"github.com/regularkevvv/agentic/harness/eval/harnesssubject",
		"github.com/regularkevvv/agentic/harness/event/inproc",
		"github.com/regularkevvv/agentic/harness/memory/jsonl",
		"github.com/regularkevvv/agentic/harness/memory/memory",
		"github.com/regularkevvv/agentic/harness/runtime/system",
		"github.com/regularkevvv/agentic/harness/skill/filesystem",
		"github.com/regularkevvv/agentic/harness/skill/memory",
		"github.com/regularkevvv/agentic/harness/store/jsonl",
		"github.com/regularkevvv/agentic/harness/store/memory",
	}
	adapterDirs := map[string]bool{
		"artifact/artifacttest": true,
		"artifact/file":         true,
		"artifact/memory":       true,
		"artifact/spill":        true,
		"codec/json":            true,
		"codemode/subprocess":   true,
		"env/envtest":           true,
		"env/local":             true,
		"env/memory":            true,
		"eval/harnesssubject":   true,
		"event/eventtest":       true,
		"event/inproc":          true,
		"memory/jsonl":          true,
		"memory/memory":         true,
		"memory/memorytest":     true,
		"runtime/system":        true,
		"skill/filesystem":      true,
		"skill/memory":          true,
		"store/jsonl":           true,
		"store/memory":          true,
		"store/storetest":       true,
	}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if relative != "." && adapterDirs[filepath.ToSlash(relative)] {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		if filepath.ToSlash(relative) == "default.go" {
			return nil
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, spec := range file.Imports {
			value, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return err
			}
			for _, prefix := range adapterPrefixes {
				if value == prefix || strings.HasPrefix(value, prefix+"/") {
					t.Errorf("%s imports adapter %q", filepath.ToSlash(relative), value)
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestHarnessModuleImportsOnlyPublicRootPackages(t *testing.T) {
	t.Parallel()
	_, filename, _, _ := runtime.Caller(0)
	root := filepath.Dir(filename)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() || filepath.Ext(path) != ".go" {
			return walkErr
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
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
				t.Errorf("%s imports root internal package %q", path, value)
			}
			return false
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
