// Command offline runs the deterministic compatible host without credentials.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/regularkevvv/agentic/tui/app"
	"github.com/regularkevvv/agentic/tui/internal/testhost"
)

func main() {
	environ := os.Environ()
	_, err := app.Run(context.Background(), testhost.New(nil), app.Options{
		Config: app.DefaultConfig(), Environ: environ, TerminalEditor: app.NewFileEditor(environ),
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
