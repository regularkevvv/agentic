package sandbox

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

func TestFactoryFailsClosedAndEnvironmentNarrows(t *testing.T) {
	var nilFactory *Factory
	if backend := nilFactory.Backend(); backend != "" {
		t.Fatalf("nil factory backend = %q", backend)
	}
	if _, err := NewFactory(Config{Root: "relative"}); err == nil {
		t.Fatal("relative sandbox root succeeded")
	}
	probeFailure := errors.New("probe failed")
	if _, err := newFactory(Config{Root: t.TempDir()}, func() (string, error) {
		return "", probeFailure
	}); !errors.Is(err, probeFailure) || !harnessenv.HasCode(err, harnessenv.CodeUnsupported) {
		t.Fatalf("backend probe failure = %v", err)
	}
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		if _, err := NewFactory(Config{Root: t.TempDir()}); err == nil {
			t.Fatal("unsupported platform silently fell back")
		}
		return
	}
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "child"), 0o755); err != nil {
		t.Fatal(err)
	}
	factory, err := NewFactory(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	if factory.Backend() == "" {
		t.Fatal("sandbox backend is empty")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := factory.Open(canceled, "canceled"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled open = %v", err)
	}
	missingRoot := filepath.Join(t.TempDir(), "missing")
	missingFactory, err := NewFactory(Config{Root: missingRoot})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := missingFactory.Open(context.Background(), "missing"); !harnessenv.HasCode(err, harnessenv.CodeNotFound) {
		t.Fatalf("missing sandbox root = %v", err)
	}
	lease, err := factory.Open(context.Background(), "session")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close(context.Background())
	narrower := lease.(harnessenv.Narrower)
	child, err := narrower.Narrow(context.Background(), harnessenv.NarrowRequest{Root: "child"})
	if err != nil {
		t.Fatal(err)
	}
	defer child.Close(context.Background())
	if _, ok := child.Shell(); ok {
		t.Fatal("narrow environment broadened shell access")
	}
	if _, err := child.(*Environment).Exec(context.Background(), harnessenv.Command{Name: "/bin/sh"}); !harnessenv.HasCode(err, harnessenv.CodeUnsupported) {
		t.Fatalf("disabled child shell = %v", err)
	}
	if _, err := narrower.Narrow(context.Background(), harnessenv.NarrowRequest{}); !harnessenv.HasCode(err, harnessenv.CodeInvalid) {
		t.Fatalf("empty narrow root = %v", err)
	}
	if _, err := narrower.Narrow(context.Background(), harnessenv.NarrowRequest{Root: "../escape"}); !harnessenv.HasCode(err, harnessenv.CodeEscaped) {
		t.Fatalf("escaped narrow root = %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "file"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := narrower.Narrow(context.Background(), harnessenv.NarrowRequest{Root: "file"}); !harnessenv.HasCode(err, harnessenv.CodeNotDirectory) {
		t.Fatalf("file narrow root = %v", err)
	}
}

func TestEnvironmentDelegatesRootedFilesystem(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("strict sandbox backend is unsupported")
	}
	root := t.TempDir()
	factory, err := NewFactory(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := factory.Open(context.Background(), "session")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close(context.Background())
	files := lease.Files()
	ctx := context.Background()
	if resource, err := files.CanonicalPath(ctx, "."); err != nil || !resource.Valid() {
		t.Fatalf("canonical root = %#v, %v", resource, err)
	}
	if err := files.MkdirAll(ctx, "nested", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := files.WriteFile(ctx, "nested/file.txt", []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}
	if data, err := files.ReadFile(ctx, "nested/file.txt"); err != nil || string(data) != "hello" {
		t.Fatalf("read = %q, %v", data, err)
	}
	if info, err := files.Stat(ctx, "nested/file.txt"); err != nil || info.Name != "file.txt" {
		t.Fatalf("stat = %#v, %v", info, err)
	}
	if entries, err := files.ReadDir(ctx, "nested"); err != nil || len(entries) != 1 || entries[0].Name != "file.txt" {
		t.Fatalf("readdir = %#v, %v", entries, err)
	}
	if err := files.Remove(ctx, "nested/file.txt"); err != nil {
		t.Fatal(err)
	}

	environment := lease.(*Environment)
	if _, err := environment.Exec(ctx, harnessenv.Command{}); !harnessenv.HasCode(err, harnessenv.CodeInvalid) {
		t.Fatalf("empty command = %v", err)
	}
	if _, err := environment.Exec(ctx, harnessenv.Command{Name: "/bin/sh", Dir: "../escape"}); !harnessenv.HasCode(err, harnessenv.CodeEscaped) {
		t.Fatalf("escaped command dir = %v", err)
	}
	if _, err := environment.Exec(ctx, harnessenv.Command{Name: "definitely-not-an-agentic-command"}); err == nil {
		t.Fatal("missing sandbox command succeeded")
	}
}

func TestSandboxAllowsExplicitNetworkPolicy(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("strict sandbox backend is unsupported")
	}
	factory, err := NewFactory(Config{Root: t.TempDir(), Network: true})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := factory.Open(context.Background(), "network")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close(context.Background())
	shell, _ := lease.Shell()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	command := harnessenv.Command{
		Name: executable,
		Args: []string{"-test.run", "^TestSandboxNetworkProbeProcess$", "-test.v"},
		Env:  []string{"AGENTIC_TEST_NETWORK_PROBE=allow"},
	}
	result, err := shell.Exec(context.Background(), command)
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("network-enabled sandbox command = %#v, %v", result, err)
	}
}

func TestSandboxResolvesRelativeWorkspaceExecutable(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("strict sandbox backend is unsupported")
	}
	root := t.TempDir()
	if err := os.Symlink("/bin/echo", filepath.Join(root, "echo")); err != nil {
		t.Fatal(err)
	}
	factory, err := NewFactory(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := factory.Open(context.Background(), "relative-executable")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close(context.Background())
	shell, _ := lease.Shell()
	result, err := shell.Exec(context.Background(), harnessenv.Command{Name: "./echo", Args: []string{"ok"}})
	if err != nil || result.ExitCode != 0 || strings.TrimSpace(string(result.Stdout)) != "ok" {
		t.Fatalf("relative workspace executable = %#v, %v", result, err)
	}
}

func TestExecutionHelpers(t *testing.T) {
	for target, want := range map[string]string{
		"/opt/go/bin/go": "/opt/go",
		"/usr/bin/git":   "/usr/libexec/git-core",
		"/bin/sh":        "/bin",
	} {
		if got := commandHelpers(target); got != want {
			t.Fatalf("command helpers for %q = %q, want %q", target, got, want)
		}
	}
	t.Setenv("AGENTIC_SANDBOX_ENV_TEST", "host")
	environment := commandEnvironment(
		[]string{"AGENTIC_SANDBOX_ENV_TEST=addition", "MALFORMED"},
		map[string]string{"AGENTIC_SANDBOX_ENV_TEST": "forced"},
	)
	found := false
	for _, entry := range environment {
		if entry == "AGENTIC_SANDBOX_ENV_TEST=forced" {
			found = true
		}
		if entry == "MALFORMED" {
			t.Fatal("malformed environment entry was retained")
		}
	}
	if !found {
		t.Fatal("forced environment value is absent")
	}

	result, err := run(context.Background(), "/bin/sh", []string{"-c", "read value; printf '%s' \"$value\"; exit 7"}, os.Environ(), t.TempDir(), []byte("stdin\n"))
	if err != nil || result.ExitCode != 7 || string(result.Stdout) != "stdin" {
		t.Fatalf("run exit result = %#v, %v", result, err)
	}
	if _, err := run(context.Background(), filepath.Join(t.TempDir(), "missing"), nil, os.Environ(), t.TempDir(), nil); err == nil {
		t.Fatal("missing executable succeeded")
	}
	relativeRoot := t.TempDir()
	if err := os.Symlink("/bin/echo", filepath.Join(relativeRoot, "relative-tool")); err != nil {
		t.Fatal(err)
	}
	previousDirectory, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(relativeRoot); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", ".")
	t.Setenv("GODEBUG", "execerrdot=0")
	resolved, resolveErr := resolveCommand("relative-tool", relativeRoot)
	if err := os.Chdir(previousDirectory); err != nil {
		t.Fatal(err)
	}
	canonicalEcho, err := filepath.EvalSymlinks("/bin/echo")
	if err != nil {
		t.Fatal(err)
	}
	if resolveErr != nil || resolved != canonicalEcho {
		t.Fatalf("relative PATH command = %q, %v", resolved, resolveErr)
	}
	path, err := privateTemp()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(path) })
	if !filepath.IsAbs(path) {
		t.Fatalf("private temp is not absolute: %q", path)
	}
	t.Setenv("TMPDIR", filepath.Join(t.TempDir(), "missing"))
	if _, err := privateTemp(); err == nil {
		t.Fatal("private temp succeeded with a missing parent")
	}
	if _, err := canonicalPrivateTemp(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("missing private temp canonicalized")
	}
	if _, err := execute(context.Background(), execution{
		root: t.TempDir(), cwd: t.TempDir(), command: harnessenv.Command{Name: "/bin/sh"},
	}); err == nil {
		t.Fatal("sandbox execution succeeded without a private temp parent")
	}
}

func TestSandboxCommandWritesWorkspaceButNotSibling(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("strict sandbox backend is unsupported")
	}
	root, outside := t.TempDir(), t.TempDir()
	factory, err := NewFactory(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := factory.Open(context.Background(), "session")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close(context.Background())
	shell, ok := lease.Shell()
	if !ok {
		t.Fatal("sandbox shell is absent")
	}
	inside, err := shell.Exec(context.Background(), harnessenv.Command{
		Name: "/bin/sh", Args: []string{"-c", "printf '%s' \"$VALUE\" > inside.txt"}, Env: []string{"VALUE=ok"},
	})
	if err != nil || inside.ExitCode != 0 {
		t.Fatalf("workspace command = %#v, %v", inside, err)
	}
	data, err := os.ReadFile(filepath.Join(root, "inside.txt"))
	if err != nil || string(data) != "ok" {
		t.Fatalf("inside write = %q, %v", data, err)
	}
	target := filepath.Join(outside, "escaped.txt")
	denied, err := shell.Exec(context.Background(), harnessenv.Command{
		Name: "/bin/sh", Args: []string{"-c", "printf no > \"$1\"", "sh", target},
	})
	if err != nil {
		t.Fatal(err)
	}
	if denied.ExitCode == 0 || (!strings.Contains(string(denied.Stderr), "denied") && !strings.Contains(string(denied.Stderr), "not permitted")) {
		t.Fatalf("outside write was not visibly denied: %#v", denied)
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Fatalf("outside target exists: %v", err)
	}
}

func TestSandboxRunsCommandSpecificToolchainChildren(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("strict sandbox backend is unsupported")
	}
	root := t.TempDir()
	for name, content := range map[string]string{
		"go.mod":        "module sandboxprobe\n\ngo 1.25\n",
		"probe_test.go": "package sandboxprobe\nimport \"testing\"\nfunc TestProbe(t *testing.T) {}\n",
	} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	factory, err := NewFactory(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := factory.Open(context.Background(), "session")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close(context.Background())
	shell, _ := lease.Shell()
	result, err := shell.Exec(context.Background(), harnessenv.Command{Name: "go", Args: []string{"test", "./..."}})
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("sandboxed go test = %#v, %v", result, err)
	}
}

func TestSandboxDeniesNetworkCreation(t *testing.T) {
	if runtime.GOOS != "darwin" && runtime.GOOS != "linux" {
		t.Skip("strict sandbox backend is unsupported")
	}
	root := t.TempDir()
	factory, err := NewFactory(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := factory.Open(context.Background(), "session")
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close(context.Background())
	shell, _ := lease.Shell()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	result, err := shell.Exec(context.Background(), harnessenv.Command{
		Name: executable,
		Args: []string{"-test.run", "^TestSandboxNetworkProbeProcess$", "-test.v"},
		Env:  []string{"AGENTIC_TEST_NETWORK_PROBE=deny"},
	})
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("network denial probe = %#v, %v", result, err)
	}
}

func TestSandboxNetworkProbeProcess(t *testing.T) {
	policy := os.Getenv("AGENTIC_TEST_NETWORK_PROBE")
	if policy == "" {
		t.Skip("sandbox child probe")
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if listener != nil {
		_ = listener.Close()
	}
	switch policy {
	case "allow":
		if err != nil {
			t.Fatalf("sandbox denied an explicitly allowed TCP listener: %v", err)
		}
	case "deny":
		if err == nil {
			t.Fatal("sandbox allowed a TCP listener")
		}
	default:
		t.Fatalf("unknown network probe policy %q", policy)
	}
}
