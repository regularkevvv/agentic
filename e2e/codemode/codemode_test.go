package codemode_test

import (
	"context"
	"os"
	"reflect"
	"testing"
	"time"

	monty "github.com/regularkevvv/gomonty"

	"github.com/regularkevvv/agentic/e2e/internal/codemodee2e"
	"github.com/regularkevvv/agentic/harness"
)

func TestHarnessCodemodeWithDownloadedGoMontyRuntime(t *testing.T) {
	if os.Getenv("GOMONTY_E2E") != "1" {
		t.Skip("set GOMONTY_E2E=1 to download the verified GoMonty runtime and run the full Harness Codemode path")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	report, err := codemodee2e.Run(ctx, codemodee2e.Options{PrepareMode: monty.PrepareDownload})
	if err != nil {
		t.Fatal(err)
	}
	if report.Runtime.RuntimeVersion == "" || report.Runtime.Target == "" || report.Runtime.Mode != monty.PrepareDownload {
		t.Fatalf("prepared runtime = %#v", report.Runtime)
	}
	if report.SessionID == "" || report.Cursor == 0 || report.State != harness.SessionIdle {
		t.Fatalf("session report = %#v", report)
	}
	if !reflect.DeepEqual(report.Capabilities, []string{"host_tools", "codemode"}) {
		t.Fatalf("capabilities = %v", report.Capabilities)
	}
	if report.ModelCalls != 2 || report.HostCalls != 1 || report.Output != codemodee2e.ExpectedOutput {
		t.Fatalf("execution report = %#v", report)
	}
}

func TestParsePrepareMode(t *testing.T) {
	t.Parallel()
	for _, value := range []string{"download", "build"} {
		mode, err := codemodee2e.ParsePrepareMode(value)
		if err != nil || string(mode) != value {
			t.Fatalf("ParsePrepareMode(%q) = %q, %v", value, mode, err)
		}
	}
	if _, err := codemodee2e.ParsePrepareMode("automatic"); err == nil {
		t.Fatal("invalid preparation mode succeeded")
	}
}
