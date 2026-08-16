package otel_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/e2e/internal/otele2e"
)

func TestCollectorSmoke(t *testing.T) {
	if os.Getenv("AGENTIC_OTEL_E2E") != "1" {
		t.Skip("set AGENTIC_OTEL_E2E=1 and run the Docker Compose stack to execute the Collector smoke proof")
	}
	endpoint := requiredEnv(t, "OTEL_EXPORTER_OTLP_ENDPOINT")
	healthURL := requiredEnv(t, "OTEL_COLLECTOR_HEALTH_URL")
	outputDirectory := requiredEnv(t, "AGENTIC_OTEL_OUTPUT_DIR")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	report, err := otele2e.Run(ctx, otele2e.Config{Endpoint: endpoint, HealthURL: healthURL})
	if err != nil {
		t.Fatal(err)
	}
	proof, err := otele2e.WaitForProof(ctx, outputDirectory)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("application scenarios: %v", report.Scenarios)
	t.Logf("Collector proof: %+v", proof)
}

func TestVerifyRejectsMissingCollectorOutput(t *testing.T) {
	_, err := otele2e.Verify(t.TempDir())
	if err == nil {
		t.Fatal("verification unexpectedly accepted a directory with no Collector exports")
	}
}

func requiredEnv(t *testing.T, name string) string {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Fatalf("%s is required", name)
	}
	if name == "AGENTIC_OTEL_OUTPUT_DIR" {
		value = filepath.Clean(value)
	}
	return value
}
