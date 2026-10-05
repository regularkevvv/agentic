package realtime

import (
	"context"
	"encoding/json"
)

// EventKind names one provider-neutral fact observed on a live call.
type EventKind string

// The facts a frontend needs from a speech-to-speech provider. Media (audio
// frames, RTP, SDP renegotiation) never appears here: it travels on whatever
// path the Conn implementation established.
const (
	EventReady               EventKind = "ready"
	EventSpeechStarted       EventKind = "speech.started"
	EventSpeechStopped       EventKind = "speech.stopped"
	EventUserTranscript      EventKind = "transcript.user"
	EventAssistantTranscript EventKind = "transcript.assistant"
	EventToolCall            EventKind = "tool.call"
	EventResponseStarted     EventKind = "response.started"
	EventResponseDone        EventKind = "response.done"
	EventError               EventKind = "error"
)

// Event is one fact received from a live call. Text carries the final
// transcript for transcript kinds, the response status for EventResponseDone,
// and the provider message for EventError. Call is present only for
// EventToolCall.
type Event struct {
	Kind EventKind
	Text string
	Call *ToolCall
}

// ToolCall is one function invocation requested by the voice model.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ActionKind names one provider-neutral instruction sent to a live call.
type ActionKind string

// The instructions a frontend issues. Configure and History are idempotent
// session setup; the rest act on the live conversation.
const (
	ActionConfigure  ActionKind = "configure"
	ActionHistory    ActionKind = "history"
	ActionUserText   ActionKind = "user.text"
	ActionToolResult ActionKind = "tool.result"
	ActionRespond    ActionKind = "respond"
	ActionCancel     ActionKind = "cancel"
)

// Action is one instruction for a live call. Exactly the payload its Kind
// names is present: Config for ActionConfigure, History for ActionHistory,
// Text for ActionUserText, Result for ActionToolResult.
type Action struct {
	Kind    ActionKind
	Config  *Config
	History []Turn
	Text    string
	Result  *ToolResult
}

// Config is the provider-neutral part of a voice session's configuration.
// Model selection, audio formats, and turn-detection tuning are deliberately
// absent: they are properties of the concrete call, fixed when it is
// established.
type Config struct {
	Instructions string
	Voice        string
	Tools        []ToolSpec
}

// ToolSpec declares one function the voice model may call. Parameters is a
// JSON Schema object.
type ToolSpec struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// ToolResult answers one ToolCall by its ID.
type ToolResult struct {
	CallID string
	Output string
}

// Role is the speaker of one history turn.
type Role string

// History roles. Tool traffic is not replayed into the voice model: the
// session owns it, and the voice model only needs what was said.
const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Turn is one prior utterance seeded into a fresh voice session.
type Turn struct {
	Role Role
	Text string
}

// Conn is the server's control channel to one live voice call. It is the
// interface every transport implements: a WebRTC call controlled through a
// provider sideband, a server-relayed WebSocket, or a SIP call all present
// the same Conn while their media flows wherever the transport put it.
//
// Send must be safe for concurrent use. Recv is called from one goroutine
// and returns an error once the call has ended or ctx is done. Close is
// idempotent and ends the server's control of the call.
type Conn interface {
	Send(context.Context, Action) error
	Recv(context.Context) (Event, error)
	Close() error
}

// Offer is a client's opaque request to establish a call, such as a WebRTC
// SDP offer or a telephony webhook payload.
type Offer struct {
	ContentType string
	Body        []byte
}

// Answer is what the client needs to complete establishment, such as the
// WebRTC SDP answer.
type Answer struct {
	ContentType string
	Body        []byte
}

// Signaler establishes calls whose media flows directly between the client
// and the provider. Accept negotiates the client's offer and returns both the
// answer for the client and the server's control Conn attached to the same
// call. Providers without a server attachment to client-originated calls
// cannot implement it; they relay media through the server and implement
// Conn directly.
type Signaler interface {
	Accept(context.Context, Offer) (Answer, Conn, error)
}
