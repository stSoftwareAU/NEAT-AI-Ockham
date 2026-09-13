#!/usr/bin/env bash
# Hermetic tests for the CI step that refreshes scripts/runlib.sh (Issue #209).
#
# The step is the only thing standing between NEAT-AI-core's `Develop` and a
# commit on this repository's PR branch, so its failure paths are worth more
# than a reading of the YAML. The step's own `run:` body is lifted out of
# `.github/workflows/ci.yml` and executed against a `gh` shim — no network, no
# token, nothing fetched — over a throwaway fixture repository.
#
# What it pins:
#   * an unfetchable file fails the step and leaves the local copy alone;
#   * a body that is not a script fails the step and leaves the copy alone;
#   * a differing, healthy copy is installed;
#   * an identical copy is left alone;
#   * a copy that is valid bash but breaks the install contract fails the step
#     — the gate that stops ungated bytes riding the job's commit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKFLOW="${REPO_ROOT}/.github/workflows/ci.yml"
CANONICAL="${SCRIPT_DIR}/runlib.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

PASSED=0
FAILED=0

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

# The `run:` body of the refresh step, dedented to column 0 — the same lines
# the runner executes.
extract_step() {
  awk '
    /^      - name: Refresh scripts\/runlib.sh from NEAT-AI-core$/ { found = 1; next }
    !found { next }
    !collecting && /^        run: \|$/ { collecting = 1; next }
    !collecting { next }
    /^[^ ]/ { exit }
    /^ {0,9}[^ ]/ { exit }
    { print substr($0, 11) }
  ' "$1"
}

STEP="${WORK_DIR}/step.sh"
extract_step "${WORKFLOW}" >"${STEP}"
if [[ ! -s "${STEP}" ]]; then
  echo "FAIL: could not extract the refresh step from ${WORKFLOW}" >&2
  exit 2
fi
bash -n "${STEP}" || {
  echo "FAIL: the extracted refresh step is not valid bash" >&2
  exit 2
}

# A `gh` that prints the file $1 instead of fetching it, or fails when $1 is
# the literal `FAIL`. Nothing here touches the network.
install_gh_shim() {
  local bin_dir="$1" payload="$2"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/gh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "${payload}" == "FAIL" ]]; then
  echo "gh: simulated fetch failure" >&2
  exit 1
fi
cat "${payload}"
EOF
  chmod +x "${bin_dir}/gh"
}

# A fixture repository: just enough for runlib.sh and its contract test to run.
new_fixture() {
  local dir="${WORK_DIR}/$1"
  mkdir -p "${dir}/scripts" "${dir}/ockham/src"
  cp "${REPO_ROOT}/Cargo.toml" "${dir}/Cargo.toml"
  cp "${REPO_ROOT}/ockham/Cargo.toml" "${dir}/ockham/Cargo.toml"
  : >"${dir}/ockham/src/main.rs"
  cp "${SCRIPT_DIR}/test-runlib.sh" "${dir}/scripts/test-runlib.sh"
  printf '#!/usr/bin/env bash\n# a stale copy\nexit 0\n' >"${dir}/scripts/runlib.sh"
  chmod +x "${dir}/scripts/runlib.sh"
  printf '%s' "${dir}"
}

run_step() {
  local dir="$1" payload="$2" name="$3"
  install_gh_shim "${WORK_DIR}/${name}-bin" "${payload}"
  (
    cd "${dir}"
    PATH="${WORK_DIR}/${name}-bin:${PATH}" \
      RUNNER_TEMP="${WORK_DIR}" \
      CORE_REF=Develop \
      CORE_REPO=stSoftwareAU/NEAT-AI-core \
      bash "${STEP}"
  ) >"${WORK_DIR}/${name}.out" 2>"${WORK_DIR}/${name}.err"
}

echo "=== an unfetchable scripts/runlib.sh fails the step ==="
DIR="$(new_fixture unfetchable)"
BEFORE="$(cksum <"${DIR}/scripts/runlib.sh")"
run_step "${DIR}" FAIL unfetchable && RC=0 || RC=$?
assert_eq "unfetchable exits non-zero" "1" "${RC}"
assert_eq "unfetchable names the repository and ref" "0" \
  "$(grep -q 'Cannot fetch scripts/runlib.sh from stSoftwareAU/NEAT-AI-core@Develop' \
    "${WORK_DIR}/unfetchable.err"; echo $?)"
assert_eq "unfetchable leaves the local copy alone" "${BEFORE}" \
  "$(cksum <"${DIR}/scripts/runlib.sh")"

echo ""
echo "=== a body that is not a script fails the step ==="
DIR="$(new_fixture notascript)"
BEFORE="$(cksum <"${DIR}/scripts/runlib.sh")"
printf '<html><body>404: Not Found</body></html>\n' >"${WORK_DIR}/notascript.payload"
run_step "${DIR}" "${WORK_DIR}/notascript.payload" notascript && RC=0 || RC=$?
assert_eq "a non-script body exits non-zero" "1" "${RC}"
assert_eq "a non-script body says so" "0" \
  "$(grep -q 'is not a bash script' "${WORK_DIR}/notascript.err"; echo $?)"
assert_eq "a non-script body leaves the local copy alone" "${BEFORE}" \
  "$(cksum <"${DIR}/scripts/runlib.sh")"

echo ""
echo "=== a differing, healthy copy is installed ==="
DIR="$(new_fixture differs)"
run_step "${DIR}" "${CANONICAL}" differs && RC=0 || RC=$?
assert_eq "a differing copy exits 0" "0" "${RC}"
assert_eq "a differing copy is overwritten byte for byte" \
  "$(cksum <"${CANONICAL}")" "$(cksum <"${DIR}/scripts/runlib.sh")"
EXECUTABLE=no
[[ -x "${DIR}/scripts/runlib.sh" ]] && EXECUTABLE=yes
assert_eq "the refreshed copy stays executable" "yes" "${EXECUTABLE}"

echo ""
echo "=== an identical copy is left alone ==="
run_step "${DIR}" "${CANONICAL}" identical && RC=0 || RC=$?
assert_eq "an identical copy exits 0" "0" "${RC}"
assert_eq "an identical copy is reported as already matching" "0" \
  "$(grep -q 'already matches' "${WORK_DIR}/identical.out"; echo $?)"

echo ""
echo "=== valid bash that breaks the install contract fails the step ==="
DIR="$(new_fixture contract)"
BEFORE_CANONICAL="$(cksum <"${CANONICAL}")"
printf '#!/usr/bin/env bash\n# parses, installs nothing\nset -euo pipefail\nexit 0\n' \
  >"${WORK_DIR}/contract.payload"
run_step "${DIR}" "${WORK_DIR}/contract.payload" contract && RC=0 || RC=$?
assert_eq "a contract-breaking copy fails the step" "1" "${RC}"
assert_eq "the failure is the contract test, not the fetch" "0" \
  "$(grep -q 'FAIL: up-to-date' "${WORK_DIR}/contract.out"; echo $?)"
assert_eq "the repository's own canonical copy is untouched by these tests" \
  "${BEFORE_CANONICAL}" "$(cksum <"${CANONICAL}")"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]
