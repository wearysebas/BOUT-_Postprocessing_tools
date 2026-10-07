#!/usr/bin/env bash
#
# Download Hermes-3 SF runs from Viking for local post-processing.
#
# Per run it fetches EVERYTHING except the raw BOUT.dmp.*.nc files:
#   BOUT.squash.nc      thinned output (make it first with squash_runs.py)
#   BOUT.restart.*.nc   final state, needed to continue / check convergence
#   BOUT.inp, BOUT.settings, BOUT.log.*, job scripts, slurm logs, ...
# plus every grid file referenced by mesh:file, into <local>/grids/.
#
# Usage:
#   ./fetch_viking_runs.sh            # preflight summary, asks before downloading
#   ./fetch_viking_runs.sh -y         # don't ask
#   ./fetch_viking_runs.sh --raw      # also fetch BOUT.dmp.*.nc for runs with no squash file
#   ./fetch_viking_runs.sh -d DIR     # local destination (default below)
#
# rsync only copies new/changed files, so re-running is cheap and resumes
# interrupted downloads.

set -euo pipefail

REMOTE="viking.york.ac.uk"
REMOTE_BASE="/users/sr1937/scratch/Hermes-Final-Pro-Max/SF-Sims"
REMOTE_ROOT="/users/sr1937/scratch/Hermes-Final-Pro-Max"
SETS=(No_Drifts CVODE_test)
LOCAL_BASE="/Users/sruiz/Dev/PhD/SF_Sims/Modified_SF_Runs/"

ASSUME_YES=0
RAW=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes) ASSUME_YES=1 ;;
        --raw) RAW=1 ;;
        -d|--dest) LOCAL_BASE="$2"; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# One SSH connection reused for every call (one 1Password / key prompt)
CM_PATH="$HOME/.ssh/cm-fetch-%r@%h:%p"
SSH="ssh -o ControlMaster=auto -o ControlPath=$CM_PATH -o ControlPersist=10m"
trap '$SSH -O exit "$REMOTE" 2>/dev/null || true' EXIT

# ---------------------------------------------------------------- preflight
echo "Scanning $REMOTE ..."
PREFLIGHT=$($SSH "$REMOTE" bash -s -- "$REMOTE_BASE" "${SETS[@]}" <<'EOF'
base="$1"; shift
for set in "$@"; do
    [[ -d "$base/$set" ]] || { echo "MISSINGSET|$set"; continue; }
    find "$base/$set" -name BOUT.inp -printf '%h\n' | sort | while read -r d; do
        sq=no; [[ -f "$d/BOUT.squash.nc" ]] && sq=yes
        nd=$(find "$d" -maxdepth 1 -name 'BOUT.dmp.*.nc' | wc -l)
        nr=$(find "$d" -maxdepth 1 -name 'BOUT.restart.*.nc' | wc -l)
        dmpsize=$(find "$d" -maxdepth 1 -name 'BOUT.dmp.*.nc' -printf '%s\n' | awk '{s+=$1} END{print s+0}')
        keepsize=$(find "$d" -type f ! -name 'BOUT.dmp.*.nc' ! -name '*.tmp' ! -name 'core.*' -printf '%s\n' | awk '{s+=$1} END{print s+0}')
        src="$d/BOUT.settings"; [[ -f "$src" ]] || src="$d/BOUT.inp"
        grid=$(awk '/^\[/{m=($0 ~ /^\[mesh\]/); next}
                    m && /^[ \t]*file[ \t]*=/{sub(/#.*/,""); sub(/^[^=]*=[ \t]*/,""); gsub(/["'"'"' \t]/,""); print; exit}' "$src")
        [[ -n "$grid" && "$grid" != /* ]] && grid="$d/$grid"
        echo "RUN|$set|${d#$base/$set/}|$sq|$nd|$nr|$dmpsize|$keepsize|$grid"
    done
done
EOF
)

total=0
declare -a GRIDS=()
declare -a WARN=()
printf '\n%-12s %-40s %-7s %5s %8s %10s\n' "set" "run" "squash" "dmps" "restarts" "download"
while IFS='|' read -r tag set run sq nd nr dmpsize keepsize grid; do
    if [[ "$tag" == MISSINGSET ]]; then WARN+=("set folder not found: $set"); continue; fi
    [[ "$tag" == RUN ]] || continue
    size=$keepsize
    if [[ "$sq" == no && "$nd" -gt 0 ]]; then
        if [[ $RAW -eq 1 ]]; then size=$((size + dmpsize))
        else WARN+=("$set/$run has no BOUT.squash.nc -> run squash_runs.py first, or use --raw ($(awk -v s="$dmpsize" 'BEGIN{printf "%.1f", s/1024^3}') GB)"); fi
    fi
    [[ "$nr" -eq 0 ]] && WARN+=("$set/$run has no restart files")
    total=$((total + size))
    [[ -n "$grid" ]] && GRIDS+=("$grid")
    printf '%-12s %-40s %-7s %5s %8s %9.2fG\n' "$set" "$run" "$sq" "$nd" "$nr" \
        "$(awk -v s="$size" 'BEGIN{print s/1024^3}')"
done <<< "$PREFLIGHT"

UNIQUE_GRIDS=()
if [[ ${#GRIDS[@]} -gt 0 ]]; then
    while IFS= read -r g; do UNIQUE_GRIDS+=("$g"); done < <(printf '%s\n' "${GRIDS[@]}" | sort -u)
fi

echo
echo "Grid files (${#UNIQUE_GRIDS[@]}):"
for g in "${UNIQUE_GRIDS[@]+"${UNIQUE_GRIDS[@]}"}"; do echo "    $g"; done
for w in "${WARN[@]+"${WARN[@]}"}"; do echo "[warn] $w"; done
echo
echo "Total to download (excluding grids): $(awk -v s="$total" 'BEGIN{printf "%.2f", s/1024^3}') GB -> $LOCAL_BASE"

if [[ $ASSUME_YES -eq 0 ]]; then
    read -r -p "Proceed? [y/N] " ans
    [[ "$ans" =~ ^[yY] ]] || { echo "Aborted."; exit 0; }
fi

# ---------------------------------------------------------------- runs
for set in "${SETS[@]}"; do
    FILTERS=(--exclude='*.tmp' --exclude='core.*')
    if [[ $RAW -eq 1 ]]; then
        while IFS='|' read -r tag s run sq nd _; do
            [[ "$tag" == RUN && "$s" == "$set" && "$sq" == no && "$nd" -gt 0 ]] || continue
            FILTERS+=(--include="/$run/BOUT.dmp.*.nc")
        done <<< "$PREFLIGHT"
    fi
    FILTERS+=(--exclude='BOUT.dmp.*.nc')

    echo; echo "=== Syncing $set"
    mkdir -p "$LOCAL_BASE/$set"
    rsync -a --partial --progress --prune-empty-dirs -e "$SSH" "${FILTERS[@]}" \
        "$REMOTE:$REMOTE_BASE/$set/" "$LOCAL_BASE/$set/"
done

# ---------------------------------------------------------------- grids
echo; echo "=== Syncing grid files"
for g in "${UNIQUE_GRIDS[@]+"${UNIQUE_GRIDS[@]}"}"; do
    if [[ "$g" == "$REMOTE_ROOT"/* ]]; then rel="${g#$REMOTE_ROOT/}"; else rel="other/$(basename "$g")"; fi
    mkdir -p "$LOCAL_BASE/grids/$(dirname "$rel")"
    rsync -a --partial -e "$SSH" "$REMOTE:$g" "$LOCAL_BASE/grids/$rel"
done

# ---------------------------------------------------------------- verify
echo; echo "=== Verifying local copies"
FAIL=0
while IFS='|' read -r tag set run sq nd nr _; do
    [[ "$tag" == RUN ]] || continue
    d="$LOCAL_BASE/$set/$run"
    lr=$(find "$d" -maxdepth 1 -name 'BOUT.restart.*.nc' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$lr" -ne "$nr" ]]; then
        echo "[fail] $set/$run: $lr of $nr restart files"; FAIL=1
    fi
done <<< "$PREFLIGHT"

python3 - "$LOCAL_BASE" "${SETS[@]}" <<'EOF' || FAIL=1
import glob, os, sys
import numpy as np
from netCDF4 import Dataset
base, sets = sys.argv[1], sys.argv[2:]
bad = 0
files = [f for s in sets for f in glob.glob(os.path.join(base, s, "**", "*.nc"), recursive=True)]
files += glob.glob(os.path.join(base, "grids", "**", "*.nc"), recursive=True)
for f in sorted(files):
    try:
        with Dataset(f) as ds:
            if f.endswith("BOUT.squash.nc"):
                if "squashoutput_version" not in ds.ncattrs():
                    raise ValueError("incomplete squash file (no squashoutput_version)")
                t = ds.variables["t_array"][:]
                if len(t) == 0 or np.ma.is_masked(t):
                    raise ValueError("incomplete squash file (t_array empty or masked)")
                print(f"    {os.path.relpath(f, base)}: {len(t)} slices, t = {float(t[0]):.4g} .. {float(t[-1]):.4g}")
    except Exception as e:
        print(f"[fail] {os.path.relpath(f, base)}: {e}")
        bad += 1
print(f"    checked {len(files) - bad}/{len(files)} netCDF files OK")
sys.exit(1 if bad else 0)
EOF

echo
if [[ $FAIL -eq 0 ]]; then echo "All done, everything verified."; else echo "Finished WITH PROBLEMS, see [fail] lines above. Re-running resumes."; exit 1; fi
