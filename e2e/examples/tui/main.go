// Command tui runs the deterministic Harness-to-TUI acceptance scenario.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/regularkevvv/agentic/e2e/internal/tuie2e"
)

func main() {
	root, err := os.MkdirTemp("", "agentic-tui-e2e-")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(root) }()
	workspace := filepath.Join(root, "workspace")
	if err := os.MkdirAll(workspace, 0o755); err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	report, err := tuie2e.Run(ctx, workspace, filepath.Join(root, "sessions"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("session=%s cursor=%d transcript=%d requests=%d cache_read=%d cache_hit=%.1f%% approvals=deny+approve marker=%q\n",
		report.SessionID, report.Cursor, report.TranscriptEntries, report.ModelRequests,
		report.CacheReadTokens, report.CacheHitPercent, report.ApprovedMarker)
}
