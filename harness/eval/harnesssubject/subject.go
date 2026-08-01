// Package harnesssubject adapts an immutable harness into the generic eval
// Subject port. Every sample owns a fresh durable session.
package harnesssubject

import (
	"context"
	"errors"
	"sync"
	"time"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness"
	evalcore "github.com/regularkevvv/agentic/harness/eval"
	"github.com/regularkevvv/agentic/harness/event"
)

type Output[O any] struct {
	Value    O                       `json:"value"`
	Messages []agentic.Message       `json:"messages"`
	Usage    agentic.Usage           `json:"usage"`
	Cursor   uint64                  `json:"cursor"`
	Status   agentic.ExecutionStatus `json:"status"`
	Events   []event.Record          `json:"events"`
}

type PromptFunc[I any] func(I) (agentic.Message, error)

type Config[I, O any] struct {
	Harness        *harness.Harness[O]
	Prompt         PromptFunc[I]
	Budget         *agentic.UsageLimits
	SessionOptions func(evalcore.Case[I]) []harness.SessionOption
	EventBuffer    int
}

type Subject[I, O any] struct{ config Config[I, O] }

func New[I, O any](config Config[I, O]) (*Subject[I, O], error) {
	if config.Harness == nil || config.Prompt == nil {
		return nil, errors.New("eval harness subject and prompt function are required")
	}
	if config.EventBuffer == 0 {
		config.EventBuffer = 4096
	}
	if config.EventBuffer < 1 {
		return nil, errors.New("eval event buffer must be positive")
	}
	if config.Budget != nil {
		copy := *config.Budget
		config.Budget = &copy
	}
	return &Subject[I, O]{config: config}, nil
}

func (s *Subject[I, O]) Run(ctx context.Context, current evalcore.Case[I]) evalcore.Outcome[Output[O]] {
	started := time.Now()
	result := evalcore.Outcome[Output[O]]{}
	prompt, err := s.config.Prompt(current.Input)
	if err != nil {
		result.Error = err
		result.ErrorMessage = err.Error()
		result.Duration = time.Since(started)
		return result
	}
	options := []harness.SessionOption(nil)
	if s.config.SessionOptions != nil {
		options = append(options, s.config.SessionOptions(current)...)
	}
	if s.config.Budget != nil {
		options = append(options, harness.WithBudget(*s.config.Budget))
	}
	session, err := s.config.Harness.NewSession(ctx, options...)
	if err != nil {
		result.Error = err
		result.ErrorMessage = err.Error()
		result.Duration = time.Since(started)
		return result
	}
	subscription := session.Subscribe(harness.SubscribeOptions{AfterCursor: 0, Buffer: s.config.EventBuffer, Preview: false})
	collector := newCollector(subscription)
	execution, runErr := session.Prompt(ctx, prompt)
	closeErr := session.Close(context.WithoutCancel(ctx))
	snapshot, snapshotErr := session.Snapshot(context.WithoutCancel(ctx))
	subscription.Close()
	events, eventErr := collector.wait()
	if execution != nil {
		result.Output.Status = execution.Status
		if execution.Result != nil {
			result.Output.Value = execution.Result.Output
		}
	}
	result.Output.Messages = snapshot.Messages
	result.Output.Usage = snapshot.Usage
	result.Output.Cursor = snapshot.Cursor
	result.Output.Events = events
	result.Error = firstError(runErr, closeErr, snapshotErr, eventErr)
	if result.Error != nil {
		result.ErrorMessage = result.Error.Error()
	}
	result.Duration = time.Since(started)
	return result
}

type collector struct {
	done   chan struct{}
	mu     sync.Mutex
	events []event.Record
	err    error
}

func newCollector(subscription *harness.Subscription) *collector {
	collector := &collector{done: make(chan struct{})}
	go func() {
		defer close(collector.done)
		events := subscription.Events
		errors := subscription.Err
		for events != nil || errors != nil {
			select {
			case record, ok := <-events:
				if !ok {
					events = nil
					continue
				}
				collector.mu.Lock()
				collector.events = append(collector.events, event.Clone(record))
				collector.mu.Unlock()
			case err, ok := <-errors:
				if !ok {
					errors = nil
					continue
				}
				collector.mu.Lock()
				collector.err = err
				collector.mu.Unlock()
			}
		}
	}()
	return collector
}

func (c *collector) wait() ([]event.Record, error) {
	<-c.done
	c.mu.Lock()
	defer c.mu.Unlock()
	values := make([]event.Record, len(c.events))
	for index, record := range c.events {
		values[index] = event.Clone(record)
	}
	return values, c.err
}

func firstError(values ...error) error {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

var _ evalcore.Subject[any, Output[any]] = (*Subject[any, any])(nil)
