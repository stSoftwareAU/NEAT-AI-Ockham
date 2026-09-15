#!/usr/bin/env bash
# Hermetic tests for the canonical scripts/runlib.sh (Issue #209).
#
# scripts/runlib.sh is a byte-identical copy of NEAT-AI-core's
# `scripts/runlib.sh` (NEAT-AI-core#680) and is never edited here — these
# tests pin the contract that copy must keep for Ockham:
#
#   * an up-to-date install runs NO cargo command at all and prints
#     `[neat_ai_ockham] already installed v<x>` on stderr, the bin path on
#     stdout;
#   * a missing or stale stamp rebuilds.
#
# A cargo shim stands in for the real toolchain and fails loud on any
# invocation the test did not expect, so no real build ever runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNLIB="${SCRIPT_DIR}/runlib.sh"
MANIFEST="${REPO_ROOT}/ockham/Cargo.toml"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
REAL_PATH="${PATH}"
REAL_HOME="${HOME}"

PASSED=0
FAILED=0

if [[ ! -x "${RUNLIB}" ]]; then
  echo "FAIL: runlib not found or not executable: ${RUNLIB}" >&2
  exit 2
fi

assert_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    echo "  PASS: ${desc}"
    PASSED=$((PASSED + 1))
  else
    echo "  FAIL: ${desc}"
    echo "    expected: '${expected}'"
    echo "    actual:   '${actual}'"
    FAILED=$((FAILED + 1))
  fi
}

# The crate version as cargo reads it — the [package] table only, so the
# `version` of a dependency can never be mistaken for the crate's own.
crate_version() {
  awk '
    /^[[:space:]]*\[/ { in_pkg = ($0 ~ /^[[:space:]]*\[package\]/) ? 1 : 0; next }
    !in_pkg { next }
    /^[[:space:]]*version[[:space:]]*=/ {
      sub(/^[^=]*=[[:space:]]*/, "")
      gsub(/"/, "")
      gsub(/[[:space:]]/, "")
      print
      exit
    }
  ' "${MANIFEST}"
}

# A cargo that answers `metadata` with the shape $1, refuses everything else
# — notably `build`, which no test here may reach by accident — and records
# every invocation in CARGO_LOG so "ran no cargo command" can be asserted on
# the calls themselves rather than on their output.
install_cargo_shim() {
  local bin_dir="$1" metadata="$2"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/cargo" <<EOF
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "\$*" >>"\${CARGO_LOG}"
if [[ "\${1:-}" == "metadata" ]]; then
  printf '%s\n' '${metadata}'
  exit 0
fi
echo "UNEXPECTED cargo: \$*" >&2
exit 99
EOF
  chmod +x "${bin_dir}/cargo"
}

VERSION="$(crate_version)"
if [[ -z "${VERSION}" ]]; then
  echo "FAIL: could not read the crate version from ${MANIFEST}" >&2
  exit 2
fi
echo "crate version under test: ${VERSION}"

METADATA="{\"packages\":[{\"name\":\"neat_ai_ockham\",\"version\":\"${VERSION}\",\"manifest_path\":\"${MANIFEST}\",\"targets\":[{\"kind\":[\"bin\"],\"name\":\"neat_ai_ockham\"}]}],\"target_directory\":\"${WORK_DIR}/target\"}"

# Every case installs into a throwaway CARGO_HOME, never the real one.
new_cargo_home() {
  local dir="${WORK_DIR}/$1/.cargo"
  mkdir -p "${dir}/bin"
  printf '%s' "${dir}"
}

stamp_install() {
  local cargo_home="$1" version="$2"
  printf 'fake\n' >"${cargo_home}/bin/neat_ai_ockham"
  chmod +x "${cargo_home}/bin/neat_ai_ockham"
  printf '%s\n' "${version}" >"${cargo_home}/bin/.neat_ai_ockham.version"
}

cd "${REPO_ROOT}"

echo ""
echo "=== up to date: no cargo command at all, bin path on stdout ==="
CARGO_HOME="$(new_cargo_home current)"
stamp_install "${CARGO_HOME}" "${VERSION}"
install_cargo_shim "${WORK_DIR}/shim" "${METADATA}"
CARGO_LOG="${WORK_DIR}/current.cargo-calls"
: >"${CARGO_LOG}"
export CARGO_HOME CARGO_LOG
HOME="${WORK_DIR}/current" PATH="${WORK_DIR}/shim:${REAL_PATH}" \
  bash "${RUNLIB}" >"${WORK_DIR}/current.out" 2>"${WORK_DIR}/current.err" && RC=0 || RC=$?
assert_eq "up-to-date exits 0" "0" "${RC}"
assert_eq "up-to-date stdout is the installed CLI path" \
  "${CARGO_HOME}/bin/neat_ai_ockham" "$(cat "${WORK_DIR}/current.out")"
assert_eq "up-to-date names the version on stderr" "0" \
  "$(grep -qF "[neat_ai_ockham] already installed v${VERSION}" "${WORK_DIR}/current.err"; echo $?)"
assert_eq "up-to-date runs no cargo command at all" "" \
  "$(cat "${CARGO_LOG}")"

echo ""
echo "=== missing stamp: rebuilds (the shim refuses the build) ==="
CARGO_HOME="$(new_cargo_home nostamp)"
stamp_install "${CARGO_HOME}" "${VERSION}"
rm -f "${CARGO_HOME}/bin/.neat_ai_ockham.version"
CARGO_LOG="${WORK_DIR}/nostamp.cargo-calls"
: >"${CARGO_LOG}"
export CARGO_HOME CARGO_LOG
HOME="${WORK_DIR}/nostamp" PATH="${WORK_DIR}/shim:${REAL_PATH}" \
  bash "${RUNLIB}" >"${WORK_DIR}/nostamp.out" 2>"${WORK_DIR}/nostamp.err" && RC=0 || RC=$?
assert_eq "missing stamp attempts a build and the shim refuses it" "99" "${RC}"
assert_eq "refused build names the unexpected cargo build" "0" \
  "$(grep -q 'UNEXPECTED cargo: build' "${WORK_DIR}/nostamp.err"; echo $?)"

echo ""
echo "=== stale stamp: a different version rebuilds ==="
CARGO_HOME="$(new_cargo_home stale)"
stamp_install "${CARGO_HOME}" "0.0.0-not-this-version"
CARGO_LOG="${WORK_DIR}/stale.cargo-calls"
: >"${CARGO_LOG}"
export CARGO_HOME CARGO_LOG
HOME="${WORK_DIR}/stale" PATH="${WORK_DIR}/shim:${REAL_PATH}" \
  bash "${RUNLIB}" >"${WORK_DIR}/stale.out" 2>"${WORK_DIR}/stale.err" && RC=0 || RC=$?
assert_eq "stale stamp attempts a build and the shim refuses it" "99" "${RC}"
assert_eq "stale stamp does not report an install" "1" \
  "$(grep -q 'already installed' "${WORK_DIR}/stale.err"; echo $?)"

echo ""
echo "=== missing binary beside a matching stamp rebuilds ==="
CARGO_HOME="$(new_cargo_home nobin)"
stamp_install "${CARGO_HOME}" "${VERSION}"
rm -f "${CARGO_HOME}/bin/neat_ai_ockham"
CARGO_LOG="${WORK_DIR}/nobin.cargo-calls"
: >"${CARGO_LOG}"
export CARGO_HOME CARGO_LOG
HOME="${WORK_DIR}/nobin" PATH="${WORK_DIR}/shim:${REAL_PATH}" \
  bash "${RUNLIB}" >"${WORK_DIR}/nobin.out" 2>"${WORK_DIR}/nobin.err" && RC=0 || RC=$?
assert_eq "missing binary attempts a build and the shim refuses it" "99" "${RC}"

unset CARGO_HOME CARGO_LOG
HOME="${REAL_HOME}"
export HOME

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]
