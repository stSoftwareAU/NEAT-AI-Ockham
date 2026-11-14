#!/usr/bin/env bash
# Replay the retired #262 smoke protocol from immutable historical source.
set -euo pipefail
umask 077

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/rust_scorer new-output-directory" >&2
  exit 2
fi
if [[ ! -f "$1" || ! -x "$1" ]]; then
  echo "scorer must be an executable file: $1" >&2
  exit 2
fi
if [[ -e "$2" || -L "$2" ]]; then
  echo "output already exists: $2" >&2
  exit 2
fi
for dependency in cargo git python3 tar timeout; do
  if ! command -v "${dependency}" >/dev/null; then
    echo "required command not found: ${dependency}" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REVISION=91e0d56ee27627382989a09dbc31b54b64f90d20
# Refuse a shallow checkout missing the revision before creating any output.
git -C "${REPOSITORY}" cat-file -e "${REVISION}^{commit}"
SCORER="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1")"
mkdir -- "$2"
OUTPUT="$(cd "$2" && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT
mkdir "${SCRATCH}/source"
git -C "${REPOSITORY}" archive "${REVISION}" | tar -x -C "${SCRATCH}/source"

# Build only archived source. The maintained optimiser has no soft-gate API.
# A separate target directory avoids interfering with any current workspace build.
(
  cd "${SCRATCH}/source"
  CARGO_TARGET_DIR="${SCRATCH}/target" cargo build --locked --release \
    -p neat_ai_ockham --example soft_gate_smoke --bin neat_ai_ockham < /dev/null
)
cp "${SCRATCH}/target/release/neat_ai_ockham" "${OUTPUT}/neat_ai_ockham"
cp "${SCORER}" "${OUTPUT}/rust_scorer"
"${SCRATCH}/target/release/examples/soft_gate_smoke" "${OUTPUT}/fixture" < /dev/null
"${OUTPUT}/neat_ai_ockham" soft-gate prepare \
  "${OUTPUT}/fixture/parent.json" "${OUTPUT}/fixture/data" \
  --scorer "${OUTPUT}/rust_scorer" --out "${OUTPUT}/manifest.json" \
  --calibration-records 64 --validation-records 128 < /dev/null \
  > "${OUTPUT}/prepare.json"

# Use freshly measured hardware/input/build pins; copy only predeclared controls.
python3 - "${SCRATCH}/source" "${OUTPUT}" "${REVISION}" <<'PY'
import hashlib
import json
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:3])
original = source / "docs/experiments/soft-gate-smoke-manifest-v1.json"
archived = json.loads(original.read_bytes())
path = output / "manifest.json"
manifest = json.loads(path.read_bytes())
for name in ("creature", "corpus"):
    if manifest[name]["sha256"] != archived[name]["sha256"]:
        raise SystemExit(f"generated {name} differs from archived smoke inputs")
for name in ("seeds", "fit", "limitsPerFamily", "independentTrialsPerBudget",
             "latencyRecords", "latencyRepetitions", "criteria", "representative",
             "dataProvenance"):
    manifest[name] = archived[name]
path.write_text(json.dumps(manifest, indent=2) + "\n")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(output / "manifest.sha256").write_text(digest + "\n")
(output / "replay.json").write_text(json.dumps({
    "formatVersion": 1,
    "experimentSourceCommit": sys.argv[3],
    "archivedManifestSha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    "replayManifestSha256": digest,
    "representative": False,
    "purpose": "protocol replay only; timings and fresh build/scorer pins can differ",
}, indent=2) + "\n")
PY

# The reviewed historical controls are now locked, before any fitting/evaluation.
MANIFEST_SHA256="$(cat "${OUTPUT}/manifest.sha256")"
timeout 21600 "${OUTPUT}/neat_ai_ockham" soft-gate run "${OUTPUT}/manifest.json" \
  --manifest-sha256 "${MANIFEST_SHA256}" --out "${OUTPUT}/study" < /dev/null \
  > "${OUTPUT}/run.json"
