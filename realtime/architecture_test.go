package realtime_test

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

const sessionloopModule = "github.com/regularkevvv/agentic/harness/sessionloop"

func moduleRoot(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate the realtime module")
	}
	return filepath.Dir(filename)
}

// TestModuleRequiresOnlySessionloop freezes the module's promise: bridging a
// voice call to a session never places Agentic, Harness, a provider SDK, or a
// WebRTC stack in the consumer's module graph. Concrete transports belong to
// the assembling application.
func TestModuleRequiresOnlySessionloop(t *testing.T) {
	t.Parallel()
	contents, err := os.ReadFile(filepath.Join(moduleRoot(t), "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(contents), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 || strings.HasPrefix(fields[0], "//") {
			continue
		}
		switch fields[0] {
		case "module", "go":
		case "require":
			if len(fields) != 3 || fields[1] != sessionloopModule {
				t.Errorf("realtime/go.mod may require only %s: %q", sessionloopModule, line)
			}
		default:
			t.Errorf("unexpected realtime/go.mod directive: %q", line)
		}
	}
}

// TestSourceImportsOnlyStandardLibraryAndSessionloop keeps provider and
// transport code out of the package. Tests may also use the sessionloop
// testkit, which lives in the same required module.
func TestSourceImportsOnlyStandardLibraryAndSessionloop(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob(filepath.Join(moduleRoot(t), "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range files {
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, spec := range file.Imports {
			imported, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				t.Fatal(err)
			}
			first := strings.SplitN(imported, "/", 2)[0]
			allowed := !strings.Contains(first, ".") ||
				imported == "github.com/regularkevvv/agentic/realtime" ||
				imported == sessionloopModule ||
				(strings.HasSuffix(path, "_test.go") && strings.HasPrefix(imported, sessionloopModule+"/"))
			if !allowed {
				t.Errorf("%s imports %q", filepath.Base(path), imported)
			}
		}
	}
}
