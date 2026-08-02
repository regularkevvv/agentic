// Package test provides test doubles and contract suites for code built on
// Agentic.
//
// It is the only package under provider/ that is not a provider. It sits there
// because everything it fakes is a provider — [TestModel] for chat, and the
// embedder, encoder, and reranker doubles beside it — and a caller reaching for
// a fake looks where the real thing lives. The capability test in the root
// package skips this directory by name for the same reason.
//
// Two kinds of thing live here. The doubles let you exercise an agent without a
// network or a key. The conformance sub-package holds the contract suite a
// provider must pass, which is what makes writing a provider outside this
// repository a bounded job rather than an exercise in reading ours.
//
// Neither is measured against the coverage threshold: a fake's branches are
// exercised by the tests that use it, and an assertion helper's failure paths
// only run when something else is broken.
package test
