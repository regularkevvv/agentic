package event

import (
	"context"
	"errors"
	"testing"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/codec"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
)

type rejectingCodec struct {
	base codec.Codec
	err  error
}

func (c rejectingCodec) Encode(any) ([]byte, error) { return nil, c.err }
func (c rejectingCodec) Decode(payload []byte, target any) error {
	return c.base.Decode(payload, target)
}

type encodingEvent struct{}

func (encodingEvent) Nature() agentic.EventNature { return agentic.EventAuthoritative }
func (encodingEvent) Type() agentic.EventType     { return agentic.EventTypeAssistantCommitted }
func (encodingEvent) TurnIndex() int              { return 0 }

func TestEventSmallValueCloneFactoryAndEncodingEdges(t *testing.T) {
	if !(EventsDropped{}).Empty() || (EventsDropped{Preview: 1}).Empty() {
		t.Fatal("EventsDropped.Empty changed")
	}
	boom := errors.New("encode")
	if _, err := FromAgentic(rejectingCodec{base: jsoncodec.New(), err: boom}, encodingEvent{}); !errors.Is(err, boom) {
		t.Fatalf("FromAgentic encoding error = %v", err)
	}
	usage := agentic.Usage{
		TotalTokens:   3,
		RequestUsages: []agentic.RequestUsage{{}},
	}
	record := Record{Payload: []byte("payload"), SessionUsed: &usage}
	cloned := Clone(record)
	cloned.Payload[0] = 'x'
	cloned.SessionUsed.TotalTokens = 9
	cloned.SessionUsed.RequestUsages = append(cloned.SessionUsed.RequestUsages, agentic.RequestUsage{})
	if string(record.Payload) != "payload" || record.SessionUsed.TotalTokens != 3 ||
		len(record.SessionUsed.RequestUsages) != 1 {
		t.Fatalf("Clone aliased record: %#v / %#v", record, cloned)
	}

	called := false
	factory := FactoryFunc(func(context.Context, []Record) (Hub, error) {
		called = true
		return nil, nil
	})
	hub, err := factory.Open(context.Background(), nil)
	if err != nil || hub != nil || !called {
		t.Fatalf("FactoryFunc = %#v, %v, called=%v", hub, err, called)
	}
	if got := (&ErrSubscriberLagged{LastCursor: 7}).Error(); got == "" {
		t.Fatal("lag error string is empty")
	}
}
