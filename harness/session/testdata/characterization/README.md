# Characterization fixtures for harness/session

These files freeze the current public behavior of `harness/session` before
the sessionloop refactor (plan `docs/design/harness-sessionloop-plan.md`,
sections 8.2, 11-S0, 12.2). They are consumed exclusively by the
`TestCharacterization*` tests in `harness/session/characterization_*.go`.

## Determinism contract

Every fixture and golden was produced with:

- `Config.Clock = fixedClock{value: 2026-01-02T03:04:05Z}` (UTC),
- `Config.IDs` = per-prefix counter returning `<prefix>_c<n>`
  (`<prefix>_r<n>` when a committed crash fixture is recovered), and
- `Config.Codec = jsoncodec.New()` — the codec identity is part of the
  fixture contract; the fixtures are only replayable under this codec.

Session IDs come from the shared test counter and are normalized to
`"SESSION_ID"` inside goldens. Store entry IDs/parents are normalized to
`e<seq>` placeholders in goldens; the `.jsonl` crash fixtures keep the real
store-generated UUIDs of the generation run.

## Regeneration

```
cd harness
go test ./session/ -run TestCharacterization -update-characterization
```

Regeneration rewrites all `*.golden.json` files and both `*.jsonl` crash
fixtures. The `.jsonl` files will differ in their random entry UUIDs on each
regeneration; payload contents stay deterministic.

## Files

- `*.golden.json` — normalized journal snapshots (`seq`, `kind`,
  `e<seq>` entry-id placeholders, codec-decoded payloads; `agentic.*`
  entries additionally have their inner event payload decoded).
- `*.events.golden.json` — the published durable event sequence
  (`name`, `nature`, `type`, `cursor`) replayed from the hub.
- `char_crash_repair.jsonl` — produced by running a complete tool-free
  prompt against a JSONL repository (journal:
  `session.created`, `run.opened`, `message`, `run.closed`), then copying
  the journal file and truncating it after the first `message` line. This
  simulates torn process loss immediately after the durable acceptance
  batch (`run.opened` + prompt) and before any driver progress. Recovery
  must close the abandoned run
  (`run.closed{interrupted, "process stopped before run termination"}`),
  open a recovery run (`Mode "continue"`, `Recovery true`), and drive
  `DriveContinue` exactly once.
- `char_crash_indeterminate.jsonl` — produced by running a complete
  single-tool exchange (tool call `crash-1`/`effect`) against a JSONL
  repository, then truncating the copied journal after the first
  `agentic.tool_started` line, leaving no `agentic.tool_result`. Recovery
  must suspend durably with kind `harness.recovery.indeterminate` and
  perform zero drives.

Tests always copy the `.jsonl` fixtures into `t.TempDir()` before `Recover`;
the committed files are never opened for writing.
