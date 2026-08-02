// Package retrieval holds the vector primitives: embedding, multi-representation
// encoding, reranking, and the vector-space identity all three are keyed on.
//
// This is one of the two halves of Agentic. The other is internal/core, which
// holds the chat primitives, and the two do not reference each other — see the
// note there. A program that only embeds text never constructs an agent, and
// the source says so.
//
// The three sub-packages are implementation detail of this one and are useful
// nowhere else:
//
//   - wire, the agentic.representations.v1 JSON contract that provider/endpoint
//     and provider/sagemaker both speak over different transports
//   - batch, which splits an oversized encoding request and reassembles it
//   - embedbatch, the dense equivalent, for providers that cap or refuse batches
//
// Nothing here is public; the root agentic package re-exports by alias.
package retrieval
