// Command main is the fresh no-replace consumer proof for the Harness release
// view. It assembles a real in-memory Harness, exposes it through the neutral
// session protocol, and proves provider streaming produces both live previews
// and authoritative committed entries.
package main

import (
	"context"
	"errors"
	"fmt"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
)

type streamingModel struct{}

func (streamingModel) Name() string { return "release:streaming-consumer" }

func (streamingModel) Request(context.Context, *agentic.ChatRequest) (*agentic.ChatResponse, error) {
	return nil, errors.New("non-streaming request used")
}

func (streamingModel) RequestStream(context.Context, *agentic.ChatRequest) (*agentic.StreamResult, error) {
	events := make(chan agentic.StreamEvent, 3)
	events <- agentic.StreamEvent{Type: agentic.StreamEventTextDelta, Delta: "stream"}
	events <- agentic.StreamEvent{Type: agentic.StreamEventTextDelta, Delta: "ed"}
	events <- agentic.StreamEvent{Type: agentic.StreamEventDone, FinishReason: agentic.FinishReasonStop}
	close(events)
	return agentic.NewStreamResult(events), nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	must(err)
	processors, err := spill.NewFactory(artifactmemory.New(), spill.Config{})
	must(err)
	runtime, err := harness.NewRuntime[string](
		agentic.NewAgent("release consumer", streamingModel{}),
		harness.RuntimeConfig{
			Sessions:         storememory.New(),
			Codec:            jsoncodec.New(),
			Events:           inproc.NewFactory(),
			Environments:     environments,
			ResultProcessors: processors,
			Clock:            system.NewClock(),
			IDs:              system.NewIDs(),
			ModelStreaming:   true,
		},
	)
	must(err)
	host, err := harness.NewSessionLoopHost(runtime)
	must(err)
	session, err := host.NewSession(ctx, sessionloop.SessionOptions{})
	must(err)
	defer func() { _ = session.Close(context.Background()) }()
	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{Preview: true, Buffer: 256})
	must(err)
	defer func() { _ = stream.Close() }()

	receipt, err := session.Dispatch(ctx, sessionloop.Command{
		Kind:  sessionloop.CommandStart,
		Input: &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}},
	})
	must(err)
	previews := 0
	previewText := ""
	entries := 0
	for {
		event, nextErr := stream.Next(ctx)
		must(nextErr)
		switch event.Kind {
		case sessionloop.EventPreviewDelta:
			if event.Preview == nil || event.Preview.Kind != sessionloop.PreviewText {
				panic(fmt.Sprintf("incompatible preview: %#v", event.Preview))
			}
			previews++
			previewText += event.Preview.Text
		case sessionloop.EventEntryCommitted:
			if event.Entry == nil {
				panic("committed entry event has no entry")
			}
			entries++
		case sessionloop.EventRunSettled:
			if event.RunID == receipt.RunID {
				if event.Outcome == nil || event.Outcome.Kind != sessionloop.RunCompleted {
					panic(fmt.Sprintf("incompatible settlement: %#v", event.Outcome))
				}
				goto settled
			}
		}
	}

settled:
	snapshot, err := session.Snapshot(ctx)
	must(err)
	if previews != 2 || previewText != "streamed" || entries != 2 || snapshot.State != sessionloop.StateIdle ||
		len(snapshot.Entries) != entries || len(snapshot.Entries[0].Blocks) != 1 ||
		len(snapshot.Entries[1].Blocks) != 1 || snapshot.Entries[0].Role != sessionloop.RoleUser ||
		snapshot.Entries[0].Blocks[0].Text != "hello" || snapshot.Entries[1].Role != sessionloop.RoleAssistant ||
		snapshot.Entries[1].Blocks[0].Text != "streamed" {
		panic(fmt.Sprintf("incompatible host: previews=%d preview_text=%q entries=%d snapshot=%#v",
			previews, previewText, entries, snapshot))
	}
	fmt.Printf("compatible_harness=%s previews=%d preview_text=%q entries=%d\n",
		snapshot.SessionID, previews, previewText, entries)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
