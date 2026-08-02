// Package core holds the chat primitives: the conversation, the model
// interface, the streaming protocol, and tool execution.
//
// This is one of the two halves of Agentic. The other is internal/retrieval,
// which holds the vector primitives, and the two do not reference each other —
// no type here mentions an embedding or a vector space, and no type there
// mentions a message or a tool. That is checked by architecture_test.go in the
// root package rather than left to habit, because the cheapest moment to
// notice the halves merging is the moment the first import is written.
//
// Nothing here is public. The root agentic package re-exports these types by
// alias so callers import one path, which is why moving a type between core
// and retrieval is a compile-time event for exactly one file, aliases.go.
package core
