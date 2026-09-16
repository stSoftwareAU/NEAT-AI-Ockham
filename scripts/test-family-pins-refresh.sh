#!/usr/bin/env bash
# Hermetic tests for the CI steps that refresh and run scripts/family-pins.sh
# (Issue #210).
#
# Ockham pins `neat-core` to a released git tag, and the only thing that moves
# that pin is the `version-increment` job: it overwrites `scripts/family-pins.sh`
# from NEAT-AI-core `Develop` and then runs it. Both steps commit to the PR
# branch, so their failure paths are worth more than a reading of the YAML. Each
# step's own `run:` body is lifted out of `.github/workflows/ci.yml` and executed
# against a `gh` shim — no network, no token, nothing fetched — over a throwaway
# fixture repository.
#
# What it pins:
#   * an unfetchable file fails the refresh and leaves the local copy alone;
#   * a body that is not a script fails the refresh and leaves the copy alone;
#   * a differing, healthy copy is installed, still executable;
#   * an identical copy is left alone;
#   * a copy that is valid bash but cannot run fails the refresh — the gate that
#     stops ungated bytes riding the job's commit;
#   * the move step reports a moved pin, and reports nothing when the pin is
#     already current.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKFLOW="${REPO_ROOT}/.github/workflows/ci.yml"
CANONICAL="${SCRIPT_DIR}/family-pins.sh"
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

# The `run:` body of the step named $2 in workflow $1, dedented to column 0 —
# the same lines the runner executes.
extract_step() {
  awk -v want="      - name: $2" '
    $0 == want { found = 1; next }
    !found { next }
    !collecting && /^        run: \|$/ { collecting = 1; next }
    !collecting { next }
    /^ {0,9}[^ ]/ { exit }
    { print substr($0, 11) }
  ' "$1"
}

extract_to() {
  local dest="$1" name="$2"
  extract_step "${WORKFLOW}" "${name}" >"${dest}"
  if [[ ! -s "${dest}" ]]; then
    echo "FAIL: could not extract the step '${name}' from ${WORKFLOW}" >&2
    exit 2
  fi
  bash -n "${dest}" || {
    echo "FAIL: the extracted step '${name}' is not valid bash" >&2
    exit 2
  }
}

REFRESH_STEP="${WORK_DIR}/refresh.sh"
MOVE_STEP="${WORK_DIR}/move.sh"
extract_to "${REFRESH_STEP}" "Refresh scripts/family-pins.sh from NEAT-AI-core"
extract_to "${MOVE_STEP}" "Move the neat-core pin to core's latest release"

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

# A fixture repository carrying a stale copy of the script being refreshed.
new_fixture() {
  local dir="${WORK_DIR}/$1"
  mkdir -p "${dir}/scripts"
  printf '#!/usr/bin/env bash\n# a stale copy\nexit 0\n' >"${dir}/scripts/family-pins.sh"
  chmod +x "${dir}/scripts/family-pins.sh"
  printf '%s' "${dir}"
}

run_refresh() {
  local dir="$1" payload="$2" name="$3"
  install_gh_shim "${WORK_DIR}/${name}-bin" "${payload}"
  (
    cd "${dir}"
    PATH="${WORK_DIR}/${name}-bin:${PATH}" \
      RUNNER_TEMP="${WORK_DIR}" \
      CORE_REF=Develop \
      CORE_REPO=stSoftwareAU/NEAT-AI-core \
      bash "${REFRESH_STEP}"
  ) >"${WORK_DIR}/${name}.out" 2>"${WORK_DIR}/${name}.err"
}

echo "=== an unfetchable scripts/family-pins.sh fails the step ==="
DIR="$(new_fixture unfetchable)"
BEFORE="$(cksum <"${DIR}/scripts/family-pins.sh")"
run_refresh "${DIR}" FAIL unfetchable && RC=0 || RC=$?
assert_eq "unfetchable exits non-zero" "1" "${RC}"
assert_eq "unfetchable names the repository and ref" "0" \
  "$(grep -q 'Cannot fetch scripts/family-pins.sh from stSoftwareAU/NEAT-AI-core@Develop' \
    "${WORK_DIR}/unfetchable.err"; echo $?)"
assert_eq "unfetchable leaves the local copy alone" "${BEFORE}" \
  "$(cksum <"${DIR}/scripts/family-pins.sh")"

echo ""
echo "=== a body that is not a script fails the step ==="
DIR="$(new_fixture notascript)"
BEFORE="$(cksum <"${DIR}/scripts/family-pins.sh")"
printf '<html><body>404: Not Found</body></html>\n' >"${WORK_DIR}/notascript.payload"
run_refresh "${DIR}" "${WORK_DIR}/notascript.payload" notascript && RC=0 || RC=$?
assert_eq "a non-script body exits non-zero" "1" "${RC}"
assert_eq "a non-script body says so" "0" \
  "$(grep -q 'is not a bash script' "${WORK_DIR}/notascript.err"; echo $?)"
assert_eq "a non-script body leaves the local copy alone" "${BEFORE}" \
  "$(cksum <"${DIR}/scripts/family-pins.sh")"

echo ""
echo "=== a differing, healthy copy is installed ==="
DIR="$(new_fixture differs)"
run_refresh "${DIR}" "${CANONICAL}" differs && RC=0 || RC=$?
assert_eq "a differing copy exits 0" "0" "${RC}"
assert_eq "a differing copy is overwritten byte for byte" \
  "$(cksum <"${CANONICAL}")" "$(cksum <"${DIR}/scripts/family-pins.sh")"
EXECUTABLE=no
[[ -x "${DIR}/scripts/family-pins.sh" ]] && EXECUTABLE=yes
assert_eq "the refreshed copy stays executable" "yes" "${EXECUTABLE}"

echo ""
echo "=== an identical copy is left alone ==="
run_refresh "${DIR}" "${CANONICAL}" identical && RC=0 || RC=$?
assert_eq "an identical copy exits 0" "0" "${RC}"
assert_eq "an identical copy is reported as already matching" "0" \
  "$(grep -q 'already matches' "${WORK_DIR}/identical.out"; echo $?)"

echo ""
echo "=== valid bash that cannot run fails the step ==="
DIR="$(new_fixture unrunnable)"
BEFORE_CANONICAL="$(cksum <"${CANONICAL}")"
printf '#!/usr/bin/env bash\n# parses, but refuses every invocation\nset -euo pipefail\nexit 3\n' \
  >"${WORK_DIR}/unrunnable.payload"
run_refresh "${DIR}" "${WORK_DIR}/unrunnable.payload" unrunnable && RC=0 || RC=$?
assert_eq "an unrunnable copy fails the step" "3" "${RC}"
assert_eq "this repository's own canonical copy is untouched by these tests" \
  "${BEFORE_CANONICAL}" "$(cksum <"${CANONICAL}")"

# --- the step that runs the script ------------------------------------------
# `family-pins.sh` is replaced by a shim so the move step is exercised without
# a network round trip: the shim rewrites the manifest exactly as the real
# script would, or leaves it alone when told the pin is current.
new_move_fixture() {
  local dir="${WORK_DIR}/$1" moves="$2"
  mkdir -p "${dir}/scripts" "${dir}/ockham"
  cat >"${dir}/scripts/family-pins.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "${moves}" == "yes" ]]; then
  sed -i 's/v1\.0\.0/v1.0.1/' ockham/Cargo.toml
  echo '[family-pins] neat-core v1.0.0 → v1.0.1 (ockham/Cargo.toml)' >&2
fi
EOF
  chmod +x "${dir}/scripts/family-pins.sh"
  printf 'neat-core = { git = "https://github.com/stSoftwareAU/NEAT-AI-core", tag = "v1.0.0" }\n' \
    >"${dir}/ockham/Cargo.toml"
  : >"${dir}/Cargo.lock"
  git -C "${dir}" init --quiet
  git -C "${dir}" add -A
  git -C "${dir}" -c user.email=t@t -c user.name=t commit --quiet -m fixture
  printf '%s' "${dir}"
}

run_move() {
  local dir="$1" name="$2"
  : >"${WORK_DIR}/${name}.env"
  (
    cd "${dir}"
    GITHUB_ENV="${WORK_DIR}/${name}.env" bash "${MOVE_STEP}"
  ) >"${WORK_DIR}/${name}.out" 2>"${WORK_DIR}/${name}.err"
}

echo ""
echo "=== a moved pin is recorded for the commit step ==="
DIR="$(new_move_fixture moved yes)"
run_move "${DIR}" moved && RC=0 || RC=$?
assert_eq "the move step exits 0" "0" "${RC}"
assert_eq "the moved pin reaches the manifest" "0" \
  "$(grep -q 'tag = "v1.0.1"' "${DIR}/ockham/Cargo.toml"; echo $?)"
assert_eq "the move is recorded in GITHUB_ENV" "0" \
  "$(grep -q '^FAMILY_PIN_MOVED=1$' "${WORK_DIR}/moved.env"; echo $?)"

echo ""
echo "=== a pin already on the latest release records nothing ==="
DIR="$(new_move_fixture current no)"
run_move "${DIR}" current && RC=0 || RC=$?
assert_eq "an unchanged pin exits 0" "0" "${RC}"
assert_eq "an unchanged pin records no move" "1" \
  "$(grep -q '^FAMILY_PIN_MOVED=1$' "${WORK_DIR}/current.env"; echo $?)"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]
