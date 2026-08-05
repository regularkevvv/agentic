// Package testkit provides a fully conforming, in-memory reference host for
// the sessionloop protocol. It doubles as the module's living documentation:
// reading this package shows how a host is expected to honor every protocol
// law without a real model, provider, or storage backend.
//
// The host advertises every standard capability except
// sessionloop.CapabilityIdempotentDispatch, mirroring the honesty rule of the
// Agentic Harness implementation: idempotency is only advertised once
// idempotency keys are durably recorded, and an in-memory map is not durable.
// WithIdempotentDispatch opts in: it advertises the capability and records
// idempotency keys on the session state itself — the same lifetime as the
// host's "durable" log — so replays return the original receipt across
// handle close/reopen without duplicate effects.
//
// Sessions are single-flight actors over an append-only in-memory log. Every
// authoritative fact is assigned a Position before the receipt returns, so
// acceptance is honestly "durable" relative to the host's own failure
// boundary (the process), and authoritative events replay strictly after any
// previously observed position. Previews carry the latest durable position,
// are dropped when a subscriber's buffer is full (counted into the next
// delivered event's Dropped field), and a full buffer on an authoritative
// event terminally fails the stream with sessionloop.ErrLagged.
//
// Runs are driven by a RunFunc configured per host. The default RunFunc
// (EchoRunFunc) emits one assistant entry echoing the start input text and
// completes. ScenarioRunFunc additionally implements the scripted scenarios
// the conformance package's Factory contract documents.
//
// HoldNextRun makes the next started run block inside the engine before its
// first step until released; the conformance suite uses it as its Gate for
// timing-dependent cases.
package testkit
