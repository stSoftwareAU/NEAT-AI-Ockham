#!/usr/bin/env bash
# Hermetic tests for the CI step that refreshes the canonical NEAT-AI-core
# scripts — scripts/runlib.sh (Issue #209) and scripts/family-pins.sh
# (Issue #210).
#
# The step is the only thing standing between NEAT-AI-core's `Develop` and a
# commit on this repository's PR branch, so its failure paths are worth more
# than a reading of the YAML. The step's own `run:` body is lifted out of
# `.github/workflows/ci.yml` and executed against a `gh` shim — no network, no
# token, nothing fetched — over a throwaway fixture repository.
#
# What it pins, for each canonical file:
#   * an unfetchable file fails the step and leaves the local copy alone;
#   * a body that is not a script fails the step and leaves the copy alone;
#   * a differing, healthy copy is installed, executable;
#   * an identical copy is left alone;
#   * a copy that is valid bash but breaks its contract — runlib.sh's install
#     contract, family-pins.sh answering `--help` — fails the step: the gate
#     that stops ungated bytes riding the job's commit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKFLOW="${REPO_ROOT}/.github/workflows/ci.yml"
CANONICAL_RUNLIB="${SCRIPT_DIR}/runlib.sh"
CANONICAL_PINS="${SCRIPT_DIR}/family-pins.sh"
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
    /^      - name: Refresh the canonical NEAT-AI-core scripts$/ { found = 1; next }
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

# A `gh` that answers a fetch of scripts/runlib.sh with the file $2 and a
# fetch of scripts/family-pins.sh with the file $3 instead of going to the
# network, or fails when the payload is the literal `FAIL`. Any other request
# is a bug in the step. Nothing here touches the network.
install_gh_shim() {
  local bin_dir="$1" runlib_payload="$2" pins_payload="$3"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/gh" <<SHIM
#!/usr/bin/env bash
set -euo pipefail
case "\${2:-}" in
  *scripts/runlib.sh*) payload="${runlib_payload}" ;;
  *scripts/family-pins.sh*) payload="${pins_payload}" ;;
  *) echo "gh shim: unexpected request: \$*" >&2; exit 1 ;;
esac
if [[ "\${payload}" == "FAIL" ]]; then
  echo "gh: simulated fetch failure" >&2
  exit 1
fi
cat "\${payload}"
SHIM
  chmod +x "${bin_dir}/gh"
}

# A fixture repository: just enough for both scripts and their contract checks
# to run, with a stale copy of each canonical file.
new_fixture() {
  local dir="${WORK_DIR}/$1"
  mkdir -p "${dir}/scripts" "${dir}/ockham/src"
  cp "${REPO_ROOT}/Cargo.toml" "${dir}/Cargo.toml"
  cp "${REPO_ROOT}/ockham/Cargo.toml" "${dir}/ockham/Cargo.toml"
  : >"${dir}/ockham/src/main.rs"
  cp "${SCRIPT_DIR}/test-runlib.sh" "${dir}/scripts/test-runlib.sh"
  for file in runlib.sh family-pins.sh; do
    printf '#!/usr/bin/env bash\n# a stale copy\nexit 0\n' >"${dir}/scripts/${file}"
    chmod +x "${dir}/scripts/${file}"
  done
  printf '%s' "${dir}"
}

run_step() {
  local dir="$1" runlib_payload="$2" pins_payload="$3" name="$4"
  install_gh_shim "${WORK_DIR}/${name}-bin" "${runlib_payload}" "${pins_payload}"
  (
    cd "${dir}"
    PATH="${WORK_DIR}/${name}-bin:${PATH}" \
      RUNNER_TEMP="${WORK_DIR}" \
      CORE_REF=Develop \
      CORE_REPO=stSoftwareAU/NEAT-AI-core \
      bash "${STEP}"
  ) >"${WORK_DIR}/${name}.out" 2>"${WORK_DIR}/${name}.err"
}

BEFORE_CANONICAL_RUNLIB="$(cksum <"${CANONICAL_RUNLIB}")"
BEFORE_CANONICAL_PINS="$(cksum <"${CANONICAL_PINS}")"

echo "=== an unfetchable scripts/runlib.sh fails the step ==="
DIR="$(new_fixture unfetchable)"
BEFORE="$(cksum <"${DIR}/scripts/runlib.sh")"
run_step "${DIR}" FAIL "${CANONICAL_PINS}" unfetchable && RC=0 || RC=$?
assert_eq "unfetchable exits non-zero" "1" "${RC}"
assert_eq "unfetchable names the file, the repository and the ref" "0" \
  "$(grep -q 'Cannot fetch scripts/runlib.sh from stSoftwareAU/NEAT-AI-core@Develop' \
    "${WORK_DIR}/unfetchable.err"; echo $?)"
assert_eq "unfetchable leaves the local copy alone" "${BEFORE}" \
  "$(cksum <"${DIR}/scripts/runlib.sh")"

echo ""
echo "=== a body that is not a script fails the step ==="
DIR="$(new_fixture notascript)"
BEFORE="$(cksum <"${DIR}/scripts/runlib.sh")"
printf '<html><body>404: Not Found</body></html>\n' >"${WORK_DIR}/notascript.payload"
run_step "${DIR}" "${WORK_DIR}/notascript.payload" "${CANONICAL_PINS}" notascript && RC=0 || RC=$?
assert_eq "a non-script body exits non-zero" "1" "${RC}"
assert_eq "a non-script body says so" "0" \
  "$(grep -q 'scripts/runlib.sh is not a bash script' "${WORK_DIR}/notascript.err"; echo $?)"
assert_eq "a non-script body leaves the local copy alone" "${BEFORE}" \
  "$(cksum <"${DIR}/scripts/runlib.sh")"

echo ""
echo "=== differing, healthy copies of both files are installed ==="
DIR="$(new_fixture differs)"
run_step "${DIR}" "${CANONICAL_RUNLIB}" "${CANONICAL_PINS}" differs && RC=0 || RC=$?
assert_eq "differing copies exit 0" "0" "${RC}"
assert_eq "a differing runlib.sh is overwritten byte for byte" \
  "$(cksum <"${CANONICAL_RUNLIB}")" "$(cksum <"${DIR}/scripts/runlib.sh")"
assert_eq "a differing family-pins.sh is overwritten byte for byte" \
  "$(cksum <"${CANONICAL_PINS}")" "$(cksum <"${DIR}/scripts/family-pins.sh")"
for file in runlib.sh family-pins.sh; do
  EXECUTABLE=no
  [[ -x "${DIR}/scripts/${file}" ]] && EXECUTABLE=yes
  assert_eq "the refreshed ${file} stays executable" "yes" "${EXECUTABLE}"
done
assert_eq "both refreshes are reported" "2" \
  "$(grep -c '^Refreshing scripts/' "${WORK_DIR}/differs.out")"

echo ""
echo "=== identical copies are left alone ==="
run_step "${DIR}" "${CANONICAL_RUNLIB}" "${CANONICAL_PINS}" identical && RC=0 || RC=$?
assert_eq "identical copies exit 0" "0" "${RC}"
assert_eq "both copies are reported as already matching" "2" \
  "$(grep -c 'already matches' "${WORK_DIR}/identical.out")"

echo ""
echo "=== valid bash that breaks the runlib install contract fails the step ==="
DIR="$(new_fixture contract)"
printf '#!/usr/bin/env bash\n# parses, installs nothing\nset -euo pipefail\nexit 0\n' \
  >"${WORK_DIR}/contract.payload"
run_step "${DIR}" "${WORK_DIR}/contract.payload" "${CANONICAL_PINS}" contract && RC=0 || RC=$?
assert_eq "a contract-breaking runlib.sh fails the step" "1" "${RC}"
assert_eq "the failure is the contract test, not the fetch" "0" \
  "$(grep -q 'FAIL: up-to-date' "${WORK_DIR}/contract.out"; echo $?)"

echo ""
echo "=== an unfetchable scripts/family-pins.sh fails the step ==="
DIR="$(new_fixture pins-unfetchable)"
BEFORE="$(cksum <"${DIR}/scripts/family-pins.sh")"
run_step "${DIR}" "${CANONICAL_RUNLIB}" FAIL pins-unfetchable && RC=0 || RC=$?
assert_eq "an unfetchable family-pins.sh exits non-zero" "1" "${RC}"
assert_eq "it names the file, the repository and the ref" "0" \
  "$(grep -q 'Cannot fetch scripts/family-pins.sh from stSoftwareAU/NEAT-AI-core@Develop' \
    "${WORK_DIR}/pins-unfetchable.err"; echo $?)"
assert_eq "it leaves the local family-pins.sh alone" "${BEFORE}" \
  "$(cksum <"${DIR}/scripts/family-pins.sh")"
assert_eq "the runlib.sh refresh that came first still happened" \
  "$(cksum <"${CANONICAL_RUNLIB}")" "$(cksum <"${DIR}/scripts/runlib.sh")"

echo ""
echo "=== valid bash that cannot answer --help fails the family-pins refresh ==="
DIR="$(new_fixture pins-contract)"
printf '#!/usr/bin/env bash\n# parses, but is not family-pins.sh\nset -euo pipefail\nexit 3\n' \
  >"${WORK_DIR}/pins-contract.payload"
run_step "${DIR}" "${CANONICAL_RUNLIB}" "${WORK_DIR}/pins-contract.payload" pins-contract && RC=0 || RC=$?
# The step exits with whatever `--help` exited — the payload's own status.
NONZERO=no
[[ "${RC}" -ne 0 ]] && NONZERO=yes
assert_eq "a contract-breaking family-pins.sh fails the step" "yes" "${NONZERO}"
assert_eq "the failure is not the fetch" "1" \
  "$(grep -q 'Cannot fetch' "${WORK_DIR}/pins-contract.err"; echo $?)"
assert_eq "the new bytes were fetched and installed before the contract check" "0" \
  "$(grep -q 'Refreshing scripts/family-pins.sh' "${WORK_DIR}/pins-contract.out"; echo $?)"

echo ""
assert_eq "the repository's own canonical runlib.sh is untouched by these tests" \
  "${BEFORE_CANONICAL_RUNLIB}" "$(cksum <"${CANONICAL_RUNLIB}")"
assert_eq "the repository's own canonical family-pins.sh is untouched by these tests" \
  "${BEFORE_CANONICAL_PINS}" "$(cksum <"${CANONICAL_PINS}")"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]
