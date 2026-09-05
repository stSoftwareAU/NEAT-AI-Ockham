## Summary

The Semgrep PR gate pinned its job container as a bare digest —
`image: semgrep/semgrep@sha256:a9ea2d56…` — with no release tag. The digest
keeps the image immutable, but Renovate's `github-actions` manager and
Dependabot's `docker` ecosystem both read the version from the **tag** and
rewrite the digest beside it, so an untagged pin is frozen forever: newer
Semgrep rule sets and security fixes would never reach this gate.

Re-pinned as `semgrep/semgrep:1.86.0@sha256:a9ea2d56…`. The digest is
unchanged, so the scanner runs the byte-for-byte identical image it ran
before; the tag simply gives updaters something to resolve future bumps from.
The surrounding comment block now documents trackability alongside the
existing immutability rationale, and the bump protocol says to move the tag,
digest and version-label comment in lockstep.

Closes #125.

## Evidence

Backend/CI change with no web interface to screenshot.

**The tag genuinely resolves to the pinned digest** — verified against the
Docker Hub registry rather than trusted from the file's own comment:

```console
$ curl -sD - -o /dev/null -H "Authorization: Bearer $TOK" \
    -H "Accept: application/vnd.oci.image.index.v1+json,…" \
    https://registry-1.docker.io/v2/semgrep/semgrep/manifests/1.86.0
HTTP/2 200
docker-content-digest: sha256:a9ea2d5621c29d815d90c2a3b2f9571da8972ef4ff855c9e4902681730240e35
```

That is the digest already in the workflow, so adding `:1.86.0` cannot change
which image runs.

**The new test was observed red before the fix and green after:**

```console
$ cargo test --test workflow_pins          # before the fix
.github/workflows/semgrep.yml:53: `semgrep/semgrep@sha256:a9ea…` pins a bare
digest with no release tag — dependency updaters resolve bumps from the tag …
test result: FAILED. 0 passed; 1 failed

$ cargo test --test workflow_pins          # after the fix
test every_container_pin_carries_a_tag_and_a_digest ... ok
```

`./quality.sh` passes end to end (shellcheck, codespell, markdownlint,
actionlint, cargo-deny, fmt, clippy, full test suite, docs): `All quality
checks passed!`

## Test Plan

- Added `ockham/tests/workflow_pins.rs::every_container_pin_carries_a_tag_and_a_digest`
  — reads every `.github/workflows/*.yml`, extracts each `image:` reference,
  and fails unless it carries both a 64-character `@sha256:` digest and a
  non-empty release tag. Written as a general contract over all workflows, so
  any future container pin is covered, not just Semgrep's; the tag split
  tolerates a registry host with a port (`host:5000/repo`).
- Existing suite unchanged and passing (10 README-contract tests, CLI,
  auto-version, real-scorer, sweep-restart).
