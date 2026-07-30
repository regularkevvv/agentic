package inproc

import (
	"testing"

	"github.com/regularkevvv/agentic/harness/event"
)

func TestCloseAndDefensiveSubscriberEdges(t *testing.T) {
	hub := New(nil)
	subscription := hub.Subscribe(event.SubscribeOptions{Buffer: 1})
	hub.Close()
	if terminal, ok := <-subscription.Err; !ok || terminal != nil {
		t.Fatalf("hub close terminal = %v, open=%v", terminal, ok)
	}

	internal := &subscriber{disconnected: true}
	if hub.deliverLocked(internal, event.Record{}) {
		t.Fatal("disconnected subscriber accepted delivery")
	}
	hub.closeLocked(internal, nil)
}
