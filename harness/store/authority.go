// Authority binds repository primitives to an externally owned execution grant.
// It protects journal writes without exposing lease renewal to harness logic.

package store

import "context"

// Authority executes a bounded storage primitive atomically with validation of
// its current grant. For a database adapter, the callback's context must carry
// the SAME transaction used for fence validation; the repository must honor it.
// A check followed by an independent write does not implement this contract.
// Invoke the callback at most once per call; ambiguous post-commit errors must
// reach the caller rather than silently rerunning resource-creating callbacks.
type Authority interface {
	Commit(context.Context, func(context.Context) error) error
}

// WithAuthority binds every create/open/load/append (including recovery and
// asynchronous execution) to the supplied grant. Both arguments are required.
// Close remains resource cleanup and must never revoke a successor's authority.
// This wrapper cannot turn unrelated storage systems into an atomic transaction.
func WithAuthority(repository Repository, authority Authority) Repository {
	if repository == nil || authority == nil {
		panic("store: repository and authority are required")
	}
	return &authorizedRepository{repository, authority}
}

type authorizedRepository struct {
	repository Repository
	authority  Authority
}

func (r *authorizedRepository) Create(ctx context.Context, id string, entries ...PendingEntry) (journal Journal, commit Commit, err error) {
	err = r.authority.Commit(ctx, func(tx context.Context) error {
		var inner error
		journal, commit, inner = r.repository.Create(tx, id, entries...)
		return inner
	})
	if err != nil {
		if journal != nil {
			_ = journal.Close(context.Background())
		}
		return nil, Commit{}, err
	}
	return &authorizedJournal{journal, r.authority}, commit, nil
}

func (r *authorizedRepository) Open(ctx context.Context, id string) (journal Journal, err error) {
	err = r.authority.Commit(ctx, func(tx context.Context) error {
		var inner error
		journal, inner = r.repository.Open(tx, id)
		return inner
	})
	if err != nil {
		if journal != nil {
			_ = journal.Close(context.Background())
		}
		return nil, err
	}
	return &authorizedJournal{journal, r.authority}, nil
}

type authorizedJournal struct {
	Journal
	authority Authority
}

func (j *authorizedJournal) Load(ctx context.Context) (snapshot Snapshot, err error) {
	err = j.authority.Commit(ctx, func(tx context.Context) error {
		var inner error
		snapshot, inner = j.Journal.Load(tx)
		return inner
	})
	if err != nil {
		return Snapshot{}, err
	}
	return snapshot, nil
}

func (j *authorizedJournal) Append(ctx context.Context, cursor Cursor, entries ...PendingEntry) (commit Commit, err error) {
	err = j.authority.Commit(ctx, func(tx context.Context) error {
		var inner error
		commit, inner = j.Journal.Append(tx, cursor, entries...)
		return inner
	})
	if err != nil {
		return Commit{}, err
	}
	return commit, nil
}
