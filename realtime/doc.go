// Package realtime is a provider-neutral voice frontend for a sessionloop
// session.
//
// A speech-to-speech model holds the call: it listens, decides when the user
// finished, speaks, and handles barge-in. It does not own the conversation.
// Everything beyond small talk is delegated through one function tool to the
// sessionloop session, which remains the single owner of execution, tools,
// approvals, and the durable transcript. The voice model's context is a
// disposable cache seeded from the session's snapshot.
//
// The package defines the interfaces (Conn, Signaler) and the business logic
// (Bridge). It names no provider and no transport. Concrete adapters — an
// OpenAI WebRTC call with a sideband control socket, an xAI WebSocket relay,
// a SIP trunk — live with the application that assembles them.
package realtime
