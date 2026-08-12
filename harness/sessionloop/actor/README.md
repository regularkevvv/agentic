# SessionLoop actors

This package turns durable SessionLoop conversations into passivating actors
without choosing an application's database, pub/sub system, tenancy model, or
provider. The durable mailbox is authoritative; `Notifier` is only a lossy
wake-up hint, and the supervisor's periodic scan recovers missed hints.

For each actor, a supervisor:

1. acquires one fenced lease;
2. activates the application-assembled SessionLoop session;
3. drains durable commands in order, one active run at a time;
4. persists observed events through the application observer; and
5. closes the live session and releases the lease once it is idle and empty.

Different actor IDs can run concurrently and on different pods. Multiple
commands for one actor remain serialized. `actor/memory` is a deterministic
single-process implementation for tests and local programs; production
applications normally implement `CommandStore` and `LeaseStore` in their
database and use NATS, Redis, or another transport only for `Notifier`.

Observers may also implement `SnapshotObserver`. The supervisor publishes the
activated session snapshot before draining commands, which lets application
projections incorporate durable recovery performed while the actor was
passive.
