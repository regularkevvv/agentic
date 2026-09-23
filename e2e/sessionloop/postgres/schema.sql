-- Reference adapter storage. The runner creates a private schema per test.
CREATE TABLE sessions (
    id text PRIMARY KEY,
    owner text,
    fence bigint NOT NULL DEFAULT 0 CHECK (fence >= 0),
    expires_at timestamptz,
    needs_execution boolean NOT NULL DEFAULT false,
    next_command_seq bigint NOT NULL DEFAULT 0 CHECK (next_command_seq >= 0),
    journal_seq bigint NOT NULL DEFAULT 0 CHECK (journal_seq >= 0),
    journal_entry_id text NOT NULL DEFAULT '',
    journal_handle text,
    last_claimed_at timestamptz,
    CHECK ((owner IS NULL) = (expires_at IS NULL))
);
CREATE TABLE commands (
    session_id text NOT NULL REFERENCES sessions(id),
    id text NOT NULL,
    sequence bigint NOT NULL CHECK (sequence > 0),
    digest bytea NOT NULL,
    payload bytea,
    accepted_at_seq bigint,
    PRIMARY KEY (session_id, id),
    UNIQUE (session_id, sequence),
    CHECK ((payload IS NULL) = (accepted_at_seq IS NOT NULL))
);
CREATE INDEX pending_commands ON commands(session_id, sequence)
    WHERE accepted_at_seq IS NULL;
CREATE TABLE journal (
    session_id text NOT NULL REFERENCES sessions(id),
    sequence bigint NOT NULL CHECK (sequence > 0),
    entry_id text NOT NULL,
    parent_id text NOT NULL,
    schema_version smallint NOT NULL,
    kind text NOT NULL,
    payload bytea NOT NULL,
    durability smallint NOT NULL,
    PRIMARY KEY (session_id, sequence),
    UNIQUE (session_id, entry_id)
);
ALTER TABLE commands ADD FOREIGN KEY (session_id, accepted_at_seq)
    REFERENCES journal(session_id, sequence);
