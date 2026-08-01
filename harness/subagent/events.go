package subagent

import (
	"context"
	"errors"
	"sync"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/event"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

type projectingFactory struct {
	base   event.Factory
	parent harnessruntime.CaptureRuntime

	mu      sync.Mutex
	active  bool
	pending []event.Record
	err     error
}

func newProjectingFactory(base event.Factory, parent harnessruntime.CaptureRuntime) *projectingFactory {
	return &projectingFactory{base: base, parent: parent}
}

func (f *projectingFactory) Open(ctx context.Context, history []event.Record) (event.Hub, error) {
	if f.base == nil {
		return nil, errors.New("subagent child event factory is required")
	}
	hub, err := f.base.Open(ctx, history)
	if err != nil {
		return nil, err
	}
	if hub == nil {
		return nil, errors.New("subagent child event factory returned nil")
	}
	return &projectingHub{Hub: hub, factory: f}, nil
}

func (f *projectingFactory) Err() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.err
}

func (f *projectingFactory) project(record event.Record) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return
	}
	if !f.active {
		f.pending = append(f.pending, event.Clone(record))
		return
	}
	f.projectLocked(record)
}

// activate makes the successfully routed child visible on the parent bus.
// Initial session events are buffered until this point, so observing the first
// child event never races ahead of addressed child control registration.
func (f *projectingFactory) activate() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return f.err
	}
	for _, record := range f.pending {
		f.projectLocked(record)
		if f.err != nil {
			break
		}
	}
	f.pending = nil
	f.active = true
	return f.err
}

func (f *projectingFactory) projectLocked(record event.Record) {
	if err := f.parent.ProjectEvent(context.Background(), record); err != nil {
		f.err = err
	}
}

type projectingHub struct {
	event.Hub
	factory *projectingFactory
}

func (h *projectingHub) PublishDurable(record event.Record) {
	h.Hub.PublishDurable(record)
	h.factory.project(event.Clone(record))
}

func (h *projectingHub) PublishPreview(record event.Record) {
	h.Hub.PublishPreview(record)
	record.Nature = agentic.EventPreview
	h.factory.project(event.Clone(record))
}

var _ event.Factory = (*projectingFactory)(nil)
var _ event.Hub = (*projectingHub)(nil)
