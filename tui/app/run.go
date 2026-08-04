package app

import (
	"context"

	tea "charm.land/bubbletea/v2"

	uit "github.com/regularkevvv/agentic/tui"
)

// Run creates and executes the terminal client and returns its final model.
func Run(ctx context.Context, host uit.Host, options Options, programOptions ...tea.ProgramOption) (*Model, error) {
	options.Context = ctx
	model, err := New(host, options)
	if err != nil {
		return nil, err
	}
	programOptions = append([]tea.ProgramOption{tea.WithContext(ctx)}, programOptions...)
	result, err := tea.NewProgram(model, programOptions...).Run()
	if err != nil {
		return nil, err
	}
	final, ok := result.(*Model)
	if !ok {
		return nil, nil
	}
	return final, nil
}
