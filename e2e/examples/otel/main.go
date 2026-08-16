// Example: credential-free Agentic telemetry exported through OTLP/gRPC.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/regularkevvv/agentic/e2e/internal/otele2e"
)

func main() {
	endpoint := flag.String("endpoint", envOr("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"), "OTLP/gRPC Collector endpoint")
	healthURL := flag.String("health-url", envOr("OTEL_COLLECTOR_HEALTH_URL", "http://localhost:13133/"), "Collector health endpoint")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	report, err := otele2e.Run(ctx, otele2e.Config{Endpoint: *endpoint, HealthURL: *healthURL})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("exported %d scenarios through OTLP/gRPC\n", len(report.Scenarios))
	for _, scenario := range report.Scenarios {
		fmt.Printf("  - %s\n", scenario)
	}
	fmt.Printf("nested=%q suspended_then_ran=%t stream=%q embeddings=%d provider_error_observed=%t\n",
		report.NestedOutput,
		report.SuspendedThenRan,
		report.StreamOutput,
		report.EmbeddingCount,
		report.ProviderError != "",
	)
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
