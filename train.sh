#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Koopman training run.
#
# TRAIN   10-15  per-step straight-start (speed sweep, turns, drive-hold)
#         17-22  per-step bent-start     (state coverage away from straight)
# HELDOUT 16, 23 long-horizon straight / curled -- rollout drift only
# EXCLUDE 1-4    teleport: ~50x MSE, pollutes the operator
#         5-7,9  legacy ramp: magnitude is per-ACTION, not mm/step; case 6 is
#                4 mm/step, above U_MAX_MM. Superseded by 10-15.
#         8      settles BEFORE recording -> near-identity transitions push
#                rho(A) toward the unit circle. 15/22 give decay data instead.
# ─────────────────────────────────────────────────────────────

DATA_ROOT="csv_timeseries_pyelastica_simple"
CSV_NAME="collected_trajectories_koopman_timeseries.csv"
SCRIPT="ngk_simple.py"
HORIZON=10

TRAIN_CASES=(1 3 4 5 6)

# Resolve a case number to its CSV path via glob, so renaming a case
# directory doesn't silently drop it from the run.
csv_for() {
  local n="$1" matches
  matches=( "${DATA_ROOT}"/case_"${n}"_*/"${CSV_NAME}" )
  if [[ ! -f "${matches[0]}" ]]; then
    echo "ERROR: no CSV for case ${n} under ${DATA_ROOT}/case_${n}_*/" >&2
    return 1
  fi
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "ERROR: case ${n} matched ${#matches[@]} directories" >&2
    return 1
  fi
  echo "${matches[0]}"
}

CSV_ARGS=()
echo "Training cases:"
for n in "${TRAIN_CASES[@]}"; do
  path="$(csv_for "$n")"
  rows=$(( $(wc -l < "$path") - 1 ))
  printf "  case %-2s  %-24s %8d rows\n" "$n" "$(basename "$(dirname "$path")")" "$rows"
  CSV_ARGS+=( "$path" )
done


python3 "${SCRIPT}" \
  --csv "${CSV_ARGS[@]}" \
  --horizon "${HORIZON}" \
  --w-stab 1e-2 \
  --w-seg 1e-4 \
  --encoded-control-dim 2 