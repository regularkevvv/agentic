import SessionContract.Flavors.Postgres.Rows
import SessionContract.Flavors.Postgres.Transactions
import SessionContract.Flavors.Postgres.Recovery
import SessionContract.Flavors.Postgres.Receipts
import SessionContract.Flavors.Postgres.Expiry
import SessionContract.Flavors.Postgres.Examples

/-! # Proved PostgreSQL transaction algorithm
Rows have an independent implementation; their meaning and replies are proved
against the session contract. Interleaved private writes, rollback, crashes and
lost replies are explicit. The theorem scope is this algorithm under the stated
storage semantics, not a verification of PostgreSQL, SQL text, drivers or Go.
-/
