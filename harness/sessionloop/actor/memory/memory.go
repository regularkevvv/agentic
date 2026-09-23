// Package memory preserves the original import path for the local-channel adapter.
// Deprecated: use actor/localchannel; this package does not implement a second store.
package memory

import "github.com/regularkevvv/agentic/harness/sessionloop/actor/localchannel"

// Store is the local-channel adapter, retained for source compatibility.
type Store = localchannel.Store

// Authority fences journal operations using the local-channel adapter's mutex.
type Authority = localchannel.Authority

// New constructs the local-channel adapter.
func New() *Store { return localchannel.New() }
