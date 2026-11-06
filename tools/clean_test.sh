#!/usr/bin/env bash
# Clean-room test of the two short notebooks, as the graders will run them.
#
# 1. Moves features/ to features_old_<date>_<time>/ and rebuilds the reference submissions
#    with src.submit (fresh feature cache).
# 2. Copies only the notebooks, requirements.txt and the raw data into $HOME/notebook_test.
# 3. Makes a fresh venv there from requirements.txt (+ nbconvert, for the test only).
# 4. Executes each notebook with nbconvert and checks its CSV's md5 against the reference.
# 5. If both match, copies the executed notebooks to the repo root and the CSVs to
#    submissions/notebook_<name>.csv.
# Ends with a short SUMMARY block. Safe to re-run.
#
# From anywhere:
#     bash "<repo>/tools/clean_test.sh"

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DATA="$REPO"
TEST="$HOME/notebook_test"
LOGS="$HOME/notebook_test_logs"
FEATURES_OLD="$REPO/features_old_$(date +%Y%m%d_%H%M)"

NAMES=("exp_038_dispatch_context" "exp_041_xgb_dispatch_context")
NOTEBOOKS=("Short notebook 1.ipynb" "Short notebook 2.ipynb")

step() { echo "[$(date +%H:%M:%S)] $*"; }

die() {  # message [log file]
    echo
    echo "==================== SUMMARY ===================="
    echo "ABORTED: $1"
    if [ -n "${2:-}" ] && [ -f "$2" ]; then
        echo "log: $2"
        echo "--- last 20 lines ---"
        tail -n 20 "$2"
    fi
    echo "================================================="
    exit 1
}

# ---------------------------------------------------------------- checks
[ -x "$REPO/.venv/bin/python3.12" ] || die "repo venv python not found: $REPO/.venv/bin/python3.12"
for nb in "${NOTEBOOKS[@]}"; do
    [ -f "$REPO/$nb" ] || die "notebook not found: $REPO/$nb (run tools/build_notebooks.py)"
done
[ -f "$REPO/requirements.txt" ] || die "not found: $REPO/requirements.txt"
for item in "data/kernel" "data/extended" "prediction_mapping.csv" "sample_submission.csv"; do
    [ -e "$SRC_DATA/$item" ] || die "raw data not found: $SRC_DATA/$item"
done

rm -rf "$LOGS"
mkdir -p "$LOGS"

# ---------------------------------------------------------------- 1. references
cd "$REPO" || die "cannot cd to $REPO"
mkdir -p "$REPO/submissions"
if [ -d "$REPO/features" ]; then
    [ ! -e "$FEATURES_OLD" ] || die "already exists: $FEATURES_OLD (wait a minute and re-run)"
    mv "$REPO/features" "$FEATURES_OLD"
    step "moved features/ to $(basename "$FEATURES_OLD")/"
fi

for name in "${NAMES[@]}"; do
    step "reference: src.submit configs/$name.yaml"
    log="$LOGS/submit_$name.log"
    /usr/bin/time -l "$REPO/.venv/bin/python" -m src.submit "configs/$name.yaml" > "$log" 2>&1 \
        || die "src.submit failed for $name" "$log"
    [ -f "$REPO/submissions/$name.csv" ] || die "reference not written: submissions/$name.csv" "$log"
done

# ---------------------------------------------------------------- 2. clean folder
step "clean folder: $TEST"
rm -rf "$TEST"
mkdir -p "$TEST"
for nb in "${NOTEBOOKS[@]}"; do
    cp "$REPO/$nb" "$TEST/"
done
cp "$REPO/requirements.txt" "$TEST/"
cp -R "$SRC_DATA/data" "$TEST/data"
cp "$SRC_DATA/prediction_mapping.csv" "$SRC_DATA/sample_submission.csv" "$TEST/"

# ---------------------------------------------------------------- 3. venv
cd "$TEST" || die "cannot cd to $TEST"
# Keep jupyter away from kernels/config in the user's home dir, like a fresh machine
export JUPYTER_DATA_DIR="$TEST/venv/share/jupyter_user"
export JUPYTER_CONFIG_DIR="$TEST/venv/etc/jupyter_user"
export PYTHONNOUSERSITE=1
unset JUPYTER_PATH PYTHONPATH

step "venv + pip install -r requirements.txt"
log="$LOGS/pip.log"
"$REPO/.venv/bin/python3.12" -m venv "$TEST/venv" > "$log" 2>&1 || die "venv creation failed" "$log"
"$TEST/venv/bin/python" -m pip install -r requirements.txt >> "$log" 2>&1 \
    || die "pip install -r requirements.txt failed" "$log"
"$TEST/venv/bin/python" -m pip install nbconvert >> "$log" 2>&1 || die "pip install nbconvert failed" "$log"

"$TEST/venv/bin/jupyter" kernelspec list --json > "$LOGS/kernelspec.json" 2>> "$log" \
    || die "jupyter kernelspec list failed" "$log"
if ! "$TEST/venv/bin/python" - "$LOGS/kernelspec.json" "$TEST/venv" >> "$log" 2>&1 <<'EOF'
import json, sys
from pathlib import Path
specs = json.load(open(sys.argv[1]))["kernelspecs"]
venv = Path(sys.argv[2]).resolve()
spec = specs.get("python3")
assert spec, "no python3 kernel"
resource_dir = Path(spec["resource_dir"]).resolve()
argv0 = spec["spec"]["argv"][0]
assert resource_dir.is_relative_to(venv), f"python3 kernel resource_dir outside venv: {resource_dir}"
# "python"/"python3" is replaced by jupyter's own (venv) interpreter at launch
assert argv0 in ("python", "python3", "python3.12") or Path(argv0).resolve().is_relative_to(venv) \
    or Path(argv0).is_relative_to(venv), f"python3 kernel runs {argv0}"
print("python3 kernel OK:", resource_dir, argv0)
EOF
then
    die "the python3 kernel is not inside $TEST/venv (see $LOGS/kernelspec.json)" "$log"
fi

# ---------------------------------------------------------------- 4–5. run + compare
STATUS=(); NB_MD5=(); REF_MD5=(); WALL=(); PEAK=(); NB_LOG=()
for i in 0 1; do
    nb="${NOTEBOOKS[$i]}"; name="${NAMES[$i]}"
    log="$LOGS/notebook_$((i + 1)).log"
    NB_LOG[$i]="$log"
    step "running: $nb"
    /usr/bin/time -l "$TEST/venv/bin/jupyter" nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "$nb" > "$log" 2>&1
    rc=$?
    WALL[$i]="$(awk '/ real /{printf "%.0f s (%.1f min)", $1, $1 / 60; exit}' "$log")"
    PEAK[$i]="$(awk '/maximum resident set size/{printf "%.2f GB", $1 / 1024 / 1024 / 1024; exit}' "$log")"
    csv="$TEST/submission_$name.csv"
    ref="$REPO/submissions/$name.csv"
    REF_MD5[$i]="$(md5 -q "$ref")"
    if [ $rc -ne 0 ]; then
        STATUS[$i]="FAIL (nbconvert exit $rc)"; NB_MD5[$i]="-"
    elif [ ! -f "$csv" ]; then
        STATUS[$i]="FAIL (no $(basename "$csv"))"; NB_MD5[$i]="-"
    else
        NB_MD5[$i]="$(md5 -q "$csv")"
        if [ "${NB_MD5[$i]}" = "${REF_MD5[$i]}" ]; then STATUS[$i]="OK"; else STATUS[$i]="FAIL (md5 differs)"; fi
    fi
done

# ---------------------------------------------------------------- 6. copy back
COPIED="no (not both OK)"
if [ "${STATUS[0]}" = "OK" ] && [ "${STATUS[1]}" = "OK" ]; then
    for i in 0 1; do
        cp "$TEST/${NOTEBOOKS[$i]}" "$REPO/"
        cp "$TEST/submission_${NAMES[$i]}.csv" "$REPO/submissions/notebook_${NAMES[$i]}.csv"
    done
    COPIED="yes: executed notebooks -> repo root, CSVs -> submissions/notebook_<name>.csv"
fi

# ---------------------------------------------------------------- 7. summary
echo
echo "==================== SUMMARY ===================="
for i in 0 1; do
    echo "${NOTEBOOKS[$i]} (${NAMES[$i]}): ${STATUS[$i]}"
    echo "  md5 notebook : ${NB_MD5[$i]}"
    echo "  md5 reference: ${REF_MD5[$i]}"
    echo "  wall time    : ${WALL[$i]:-?}"
    echo "  peak memory  : ${PEAK[$i]:-?} (max RSS)"
    echo "  log          : ${NB_LOG[$i]}"
done
echo "copied back: $COPIED"
echo "other logs : $LOGS (submit_*.log, pip.log, kernelspec.json)"
for i in 0 1; do
    if [ "${STATUS[$i]}" != "OK" ]; then
        echo "--- last 20 lines of ${NB_LOG[$i]} ---"
        tail -n 20 "${NB_LOG[$i]}"
    fi
done
echo "================================================="
[ "${STATUS[0]}" = "OK" ] && [ "${STATUS[1]}" = "OK" ]
