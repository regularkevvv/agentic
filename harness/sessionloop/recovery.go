package sessionloop

import "context"

// RecoveryCloser stops the live handle without canceling accepted commands or
// declaring a run settled. It joins local execution before releasing resources;
// the journal remains reconstructible. A cleanup timeout must leave authority
// and unfinished work intact. Actor-capable sessions must implement this port.
type RecoveryCloser interface {
	Abandon(context.Context) error
}
