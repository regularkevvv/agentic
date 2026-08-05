// Command main is the fresh no-replace consumer proof for the sessionloop
// release view: a standard-library-only program that dispatches a start
// command against the testkit reference host, observes the receipt, ordered
// authoritative events, and the reconciling snapshot, then exits 0. Its
// module graph must contain nothing beyond the sessionloop module itself —
// no Agentic, Harness, TUI, provider SDK, or terminal dependency.
package main

import (
	"context"
	"fmt"
	"time"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	host := testkit.New()
	session, err := host.NewSession(ctx, sessionloop.SessionOptions{})
	if err != nil {
		panic(err)
	}
	defer func() { _ = session.Close(context.Background()) }()

	stream, err := session.Subscribe(ctx, sessionloop.SubscribeOptions{})
	if err != nil {
		panic(err)
	}
	defer func() { _ = stream.Close() }()

	receipt, err := session.Dispatch(ctx, sessionloop.Command{
		Kind:  sessionloop.CommandStart,
		Input: &sessionloop.Input{Content: []sessionloop.Block{{Kind: sessionloop.BlockText, Text: "consumer proof"}}},
	})
	if err != nil {
		panic(err)
	}
	if receipt.RunID == "" || receipt.Guarantee != sessionloop.AcceptanceDurable || receipt.Position.IsZero() {
		panic(fmt.Sprintf("incompatible receipt: %#v", receipt))
	}

	entries := 0
	for {
		event, err := stream.Next(ctx)
		if err != nil {
			panic(err)
		}
		if event.Kind == sessionloop.EventEntryCommitted {
			entries++
		}
		if event.Kind == sessionloop.EventRunSettled && event.RunID == receipt.RunID {
			if event.Outcome == nil || event.Outcome.Kind != sessionloop.RunCompleted {
				panic(fmt.Sprintf("incompatible settlement: %#v", event))
			}
			break
		}
	}

	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		panic(err)
	}
	if snapshot.SessionID != session.ID() || snapshot.State != sessionloop.StateIdle || len(snapshot.Entries) != entries || entries < 2 {
		panic(fmt.Sprintf("incompatible snapshot: %#v", snapshot))
	}

	fmt.Printf("compatible_host=%s state=%s run=%s entries=%d\n", snapshot.SessionID, snapshot.State, receipt.RunID, entries)
}

var _ sessionloop.Host = (*testkit.Host)(nil)
