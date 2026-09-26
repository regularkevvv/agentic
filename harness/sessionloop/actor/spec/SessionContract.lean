import SessionContract.Model
import SessionContract.Safety
import SessionContract.Recovery
import SessionContract.Adapter
import SessionContract.Flavors.Receive
import SessionContract.Flavors.Postgres
import SessionContract.Progress
import SessionContract.Examples
import SessionContract.Isolation
import SessionContract.Publication
import SessionContract.PublicationExamples
import SessionContract.RecoveryStartup
import SessionContract.RecoveryFrontier
import SessionContract.Cleanup

/-! # Session delivery contract
The verification root imports the executable model, universal proofs, adapter
obligations, receive/PostgreSQL flavors, publication ordering and regression
examples. Building this module checks the complete specification; Audit.lean
separately audits its proof dependencies.
-/
