// Mailbox stores immutable pending envelopes. Acknowledgment leaves only a
// digest/sequence/journal-reference tombstone; the payload is not retained twice.

package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"math"

	"github.com/jackc/pgx/v5"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/harness/sessionloop/actor"
	"github.com/regularkevvv/agentic/harness/store"
)

func (s *Store) Submit(ctx context.Context, command actor.Command) (actor.Submission, error) {
	c, err := command.Normalize()
	if err != nil {
		return actor.Submission{}, err
	}
	identity, err := json.Marshal(c.Command)
	if err != nil {
		return actor.Submission{}, err
	}
	digest := sha256.Sum256(identity)
	payload, err := encodeCommand(c.Command)
	if err != nil {
		return actor.Submission{}, err
	}
	result := actor.Submission{ID: c.ID, ActorID: c.ActorID, Guarantee: s.Guarantee()}
	err = s.transaction(ctx, func(tx pgx.Tx) error {
		// Require the explicitly created journal binding. Never silently create a
		// new journal when a referenced session is missing or damaged.
		r, err := lock(ctx, tx, c.ActorID)
		if err != nil {
			return err
		}
		var original []byte
		var seq int64
		err = tx.QueryRow(ctx, "SELECT digest, sequence FROM commands WHERE session_id=$1 AND id=$2", string(c.ActorID), string(c.ID)).Scan(&original, &seq)
		if err == nil {
			if !bytes.Equal(original, digest[:]) {
				return actor.ErrCommandConflict
			}
			result.Sequence, result.Duplicate = uint64(seq), true
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		if r.commandSeq == math.MaxInt64 {
			return actor.ErrGenerationExhausted
		}
		seq = r.commandSeq + 1
		if _, err := tx.Exec(ctx, "INSERT INTO commands(session_id,id,sequence,digest,payload) VALUES($1,$2,$3,$4,$5)", string(c.ActorID), string(c.ID), seq, digest[:], payload); err != nil {
			return err
		}
		_, err = tx.Exec(ctx, "UPDATE sessions SET next_command_seq=$2 WHERE id=$1", string(c.ActorID), seq)
		result.Sequence = uint64(seq)
		return err
	})
	if err != nil {
		return actor.Submission{}, err
	}
	return result, nil
}

func (s *Store) Pending(ctx context.Context, lease actor.Lease, after uint64, limit int) ([]actor.Command, error) {
	var result []actor.Command
	err := s.transaction(ctx, func(tx pgx.Tx) error {
		if _, err := owned(ctx, tx, lease); err != nil {
			return err
		}
		if after > math.MaxInt64 {
			return nil
		}
		if limit <= 0 {
			limit = math.MaxInt32
		}
		rows, err := tx.Query(ctx, `SELECT id,sequence,payload,digest FROM commands WHERE session_id=$1
			AND accepted_at_seq IS NULL AND sequence>$2 ORDER BY sequence LIMIT $3`, string(lease.ActorID), int64(after), limit)
		if err != nil {
			return err
		}
		defer rows.Close()
		for rows.Next() {
			c := actor.Command{ActorID: lease.ActorID}
			var seq int64
			var payload, digest []byte
			if err := rows.Scan(&c.ID, &seq, &payload, &digest); err != nil {
				return err
			}
			if err := decodeCommand(payload, &c.Command); err != nil {
				return err
			}
			identity, err := json.Marshal(c.Command)
			if err != nil {
				return err
			}
			actual := sha256.Sum256(identity)
			if c.Command.ID != sessionloop.CommandID(c.ID) || !bytes.Equal(actual[:], digest) {
				return store.ErrCorruptLog
			}
			c.Sequence = uint64(seq)
			result = append(result, c)
		}
		return rows.Err()
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Acknowledge is a runtime port, not a client API. The shared Worker has already
// compared this receipt with Session.Acceptance for the exact normalized input.
// Here its durable journal reference and the current grant are checked atomically
// with payload removal; this adapter does not decode Harness-private payloads.
func (s *Store) Acknowledge(ctx context.Context, lease actor.Lease, id actor.CommandID, receipt sessionloop.Receipt) error {
	return s.transaction(ctx, func(tx pgx.Tx) error {
		if _, err := owned(ctx, tx, lease); err != nil {
			return err
		}
		if receipt.CommandID != sessionloop.CommandID(id) || receipt.SessionID != sessionloop.SessionID(lease.ActorID) ||
			receipt.Guarantee != sessionloop.AcceptanceDurable || receipt.Position.Sequence == 0 ||
			receipt.Position.Sequence > math.MaxInt64 || receipt.Position.Token == "" {
			return actor.ErrInvalidReceipt
		}
		var accepted *int64
		if err := tx.QueryRow(ctx, "SELECT accepted_at_seq FROM commands WHERE session_id=$1 AND id=$2", string(lease.ActorID), string(id)).Scan(&accepted); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return actor.ErrCommandNotFound
			}
			return err
		}
		seq := int64(receipt.Position.Sequence)
		var exists bool
		if err := tx.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM journal WHERE session_id=$1 AND sequence=$2 AND entry_id=$3)", string(lease.ActorID), seq, receipt.Position.Token).Scan(&exists); err != nil {
			return err
		}
		if !exists || (accepted != nil && *accepted != seq) {
			return actor.ErrInvalidReceipt
		}
		_, err := tx.Exec(ctx, "UPDATE commands SET payload=NULL, accepted_at_seq=$3 WHERE session_id=$1 AND id=$2", string(lease.ActorID), string(id), seq)
		return err
	})
}
