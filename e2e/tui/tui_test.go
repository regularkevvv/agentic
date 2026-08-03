package tui_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/e2e/internal/tuie2e"
)

func TestHarnessTUIEndToEnd(t *testing.T) {
	workspace := t.TempDir()
	sessions := filepath.Join(t.TempDir(), "sessions")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	report, err := tuie2e.Run(ctx, workspace, sessions)
	if err != nil {
		t.Fatal(err)
	}
	if report.DeniedMarkerAbsent != true || report.ApprovedMarker != "approved" || report.SessionID == "" || report.Cursor == 0 {
		t.Fatalf("runtime report = %#v", report)
	}
	if report.ModelRequests != 4 || !report.StableCacheKeys || !report.AppendOnlyPrefixes || report.CacheHitPercent <= 0 {
		t.Fatalf("cache report = %#v", report)
	}
}
