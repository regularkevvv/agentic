package harness

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	artifactcapability "github.com/regularkevvv/agentic/harness/capability/artifacts"
	"github.com/regularkevvv/agentic/harness/contextpolicy"
)

func TestAssembleDefaultIsPublicCapabilityComposition(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	workspace := filepath.Join(root, "workspace")
	if err := os.Mkdir(workspace, 0o755); err != nil {
		t.Fatal(err)
	}
	config := DefaultConfig{
		WorkspaceRoot:       workspace,
		SessionDir:          filepath.Join(root, "sessions"),
		ContextWindowTokens: 128_000,
	}
	assembly, err := AssembleDefault(config)
	if err != nil {
		t.Fatal(err)
	}
	if assembly.Runtime.ToolSummarizer == nil || assembly.Runtime.ToolSummarizer(agentic.ToolUse{
		Name: "run_command", Input: map[string]any{"name": "go", "args": []any{"test", "./..."}},
	}) != "go test ./..." {
		t.Fatal("default environment tool summarizer is not installed")
	}
	reconstructed, err := New(
		&facadeDriver{},
		WithRuntime(assembly.Runtime),
		WithCapabilities(assembly.Capabilities...),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"environment", "artifacts", "contextpolicy", "permissions"}
	if !reflect.DeepEqual(reconstructed.Capabilities(), want) {
		t.Fatalf("reconstructed capabilities = %v", reconstructed.Capabilities())
	}
	session, err := reconstructed.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, directory := range []string{
		filepath.Join(config.SessionDir, "journals"),
		filepath.Join(config.SessionDir, "artifacts"),
	} {
		info, err := os.Stat(directory)
		if err != nil || !info.IsDir() {
			t.Fatalf("default directory %s: info=%#v err=%v", directory, info, err)
		}
	}

	direct, err := Default(&facadeDriver{}, DefaultConfig{
		WorkspaceRoot:       workspace,
		SessionDir:          filepath.Join(root, "other-sessions"),
		ContextWindowTokens: 128_000,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(direct.Capabilities(), want) {
		t.Fatalf("Default capabilities = %v", direct.Capabilities())
	}
}

func TestDefaultRejectsImplicitOrOverlappingPathsAndInvalidGeometry(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	workspace := filepath.Join(root, "workspace")
	if err := os.Mkdir(workspace, 0o755); err != nil {
		t.Fatal(err)
	}
	tests := map[string]DefaultConfig{
		"relative workspace": {
			WorkspaceRoot:       "workspace",
			SessionDir:          filepath.Join(root, "sessions"),
			ContextWindowTokens: 1000,
		},
		"relative sessions": {
			WorkspaceRoot:       workspace,
			SessionDir:          "sessions",
			ContextWindowTokens: 1000,
		},
		"overlap": {
			WorkspaceRoot:       workspace,
			SessionDir:          filepath.Join(workspace, ".sessions"),
			ContextWindowTokens: 1000,
		},
		"zero window": {
			WorkspaceRoot: workspace,
			SessionDir:    filepath.Join(root, "sessions"),
		},
		"negative grace": {
			WorkspaceRoot:         workspace,
			SessionDir:            filepath.Join(root, "sessions-grace"),
			ContextWindowTokens:   1000,
			ToolCancellationGrace: -1,
		},
		"ambiguous disabled spill": {
			WorkspaceRoot:        workspace,
			SessionDir:           filepath.Join(root, "sessions-spill"),
			ContextWindowTokens:  1000,
			DisableArtifactSpill: true,
			SpillThreshold:       10,
		},
	}
	for name, config := range tests {
		name, config := name, config
		t.Run(name, func(t *testing.T) {
			if _, err := AssembleDefault(config); err == nil {
				t.Fatal("AssembleDefault succeeded")
			}
		})
	}
}

func TestDefaultPathCanonicalizationAdapterFailuresAndHelpers(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	workspace := filepath.Join(root, "workspace")
	if err := os.Mkdir(workspace, 0o755); err != nil {
		t.Fatal(err)
	}
	workspaceFile := filepath.Join(root, "workspace-file")
	if err := os.WriteFile(workspaceFile, []byte("file"), 0o600); err != nil {
		t.Fatal(err)
	}
	for name, config := range map[string]DefaultConfig{
		"missing workspace": {
			WorkspaceRoot:       filepath.Join(root, "missing"),
			SessionDir:          filepath.Join(root, "sessions-missing"),
			ContextWindowTokens: 1000,
		},
		"workspace file": {
			WorkspaceRoot:       workspaceFile,
			SessionDir:          filepath.Join(root, "sessions-file"),
			ContextWindowTokens: 1000,
		},
		"invalid spill": {
			WorkspaceRoot:       workspace,
			SessionDir:          filepath.Join(root, "sessions-spill-invalid"),
			ContextWindowTokens: 1000,
			SpillThreshold:      10,
			SpillHead:           8,
			SpillTail:           8,
		},
	} {
		config := config
		t.Run(name, func(t *testing.T) {
			if _, err := AssembleDefault(config); err == nil {
				t.Fatal("invalid default succeeded")
			}
		})
	}

	sessionFile := filepath.Join(root, "session-file")
	if err := os.WriteFile(sessionFile, []byte("file"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := AssembleDefault(DefaultConfig{
		WorkspaceRoot:       workspace,
		SessionDir:          sessionFile,
		ContextWindowTokens: 1000,
	}); err == nil {
		t.Fatal("session file succeeded")
	}

	sessionDir := filepath.Join(root, "sessions-artifact")
	if err := os.Mkdir(sessionDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sessionDir, "artifacts"), []byte("file"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := AssembleDefault(DefaultConfig{
		WorkspaceRoot:       workspace,
		SessionDir:          sessionDir,
		ContextWindowTokens: 1000,
	}); err == nil {
		t.Fatal("artifact file succeeded")
	}

	existing, err := canonicalFuturePath(sessionDir)
	wantExisting, evalErr := filepath.EvalSymlinks(sessionDir)
	if err != nil || evalErr != nil || existing != wantExisting {
		t.Fatalf("existing canonical path = %q, %v", existing, err)
	}
	future := filepath.Join(root, "future", "nested")
	canonical, err := canonicalFuturePath(future)
	canonicalRoot, evalErr := filepath.EvalSymlinks(root)
	wantFuture := filepath.Join(canonicalRoot, "future", "nested")
	if err != nil || evalErr != nil || canonical != wantFuture {
		t.Fatalf("future canonical path = %q, %v", canonical, err)
	}
	if !pathsOverlap(workspace, filepath.Join(workspace, "child")) ||
		pathsOverlap(workspace, filepath.Join(root, "workspace-sibling")) ||
		!pathWithin(workspace, workspace) {
		t.Fatal("path overlap helpers returned an invalid relation")
	}
	if defaultSummaryBytes(100) != 256 ||
		defaultSummaryBytes(4*contextpolicy.DefaultStructuredSummaryBytes+1) != contextpolicy.DefaultStructuredSummaryBytes {
		t.Fatal("default summary bounds changed")
	}
	if artifactReadLimit(DefaultConfig{DisableArtifactSpill: true}) != artifactcapability.DefaultReadBytes ||
		artifactReadLimit(DefaultConfig{SpillThreshold: 1}) != 1 {
		t.Fatal("artifact read bounds changed")
	}
}

func TestDefaultValidatesDriverBeforeFilesystemMutation(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	workspace := filepath.Join(root, "workspace")
	if err := os.Mkdir(workspace, 0o755); err != nil {
		t.Fatal(err)
	}
	sessionDir := filepath.Join(root, "must-not-exist")
	if _, err := Default[string](runnerOnly{}, DefaultConfig{
		WorkspaceRoot:       workspace,
		SessionDir:          sessionDir,
		ContextWindowTokens: 1000,
	}); err == nil {
		t.Fatal("runner-only Default succeeded")
	}
	if _, err := os.Stat(sessionDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Default mutated filesystem before driver validation: %v", err)
	}
}

func TestDefaultPropagatesAssemblyValidationAfterDriverCheck(t *testing.T) {
	if _, err := Default(&facadeDriver{}, DefaultConfig{}); err == nil {
		t.Fatal("Default accepted an invalid assembly config")
	}
}

func TestDefaultRejectsSessionDirectorySymlinkLoop(t *testing.T) {
	root := t.TempDir()
	workspace := filepath.Join(root, "workspace")
	if err := os.Mkdir(workspace, 0o755); err != nil {
		t.Fatal(err)
	}
	first := filepath.Join(root, "first")
	second := filepath.Join(root, "second")
	if err := os.Symlink(filepath.Base(second), first); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Base(first), second); err != nil {
		t.Fatal(err)
	}
	if _, err := AssembleDefault(DefaultConfig{
		WorkspaceRoot:       workspace,
		SessionDir:          first,
		ContextWindowTokens: 1_000,
	}); err == nil {
		t.Fatal("session-directory symlink loop succeeded")
	}
}
