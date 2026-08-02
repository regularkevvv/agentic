// Example: a complete Harness Codemode run backed by a verified GoMonty
// release runtime and a real Monty worker subprocess.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/regularkevvv/agentic/e2e/internal/codemodee2e"
)

func main() {
	prepare := flag.String("prepare", "download", "GoMonty runtime preparation mode: download or build")
	timeout := flag.Duration("timeout", 3*time.Minute, "maximum preparation and execution time")
	flag.Parse()

	mode, err := codemodee2e.ParsePrepareMode(*prepare)
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	report, err := codemodee2e.Run(ctx, codemodee2e.Options{PrepareMode: mode})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("GoMonty runtime: %s (%s, prepared by %s)\n", report.Runtime.RuntimeVersion, report.Runtime.Target, report.Runtime.Mode)
	fmt.Printf("Harness session: %s (durable cursor %d, state %s)\n", report.SessionID, report.Cursor, report.State)
	fmt.Printf("Capabilities: %s\n", strings.Join(report.Capabilities, " -> "))
	fmt.Printf("Calls: model=%d selected_tool=%d\n", report.ModelCalls, report.HostCalls)
	fmt.Printf("Result: %s\n", report.Output)
}
