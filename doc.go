// Package agentic builds AI agents in Go: tool use, structured output,
// streaming, resumable execution, and multi-agent delegation, over a provider
// interface rather than a vendor SDK.
//
// # The two halves
//
// Agentic covers two problems that share providers but nothing else. Chat is
// the agent loop — [Agent] sends a conversation to a [Model], executes the
// tools the model asks for, and folds the results back until the model stops.
// Retrieval is the vector side — [Embedder] turns text into dense vectors,
// [RepresentationEncoder] returns dense, learned sparse, and token multi-vector
// forms from one call, and [Reranker] scores query-document pairs directly.
//
// Nothing on one side depends on the other. A program that only embeds never
// constructs an agent, and the split is real in the source: the chat types live
// in internal/core and the retrieval types in internal/retrieval, with no
// reference between them. This package re-exports both so that callers import
// one path, and the aliases are the only place the halves meet.
//
// # Providers
//
// Every provider lives under provider/ and declares what it implements with a
// compile-time assertion, because capability is not one-to-one: bedrock is a
// model and an embedder, cohere an embedder and a reranker. The README carries
// the table of which provider does what, and a test in this package fails if a
// provider stops declaring it.
//
// # Getting started
//
//	model, err := openai.New("gpt-4o")
//	agent := agentic.NewAgent("You are a helpful assistant.", model)
//	result, err := agent.Run(ctx, "What is the capital of France?")
//
// See ARCHITECTURE.md in the repository for where each kind of code lives, and
// the README for worked examples of each feature.
package agentic
