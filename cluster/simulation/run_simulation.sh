#!/usr/bin/env bash
set -uo pipefail
n="$1"; p="$2"; start="$3"; count="$4"; mode="$5"
stem="sepm_sim_n${n}_p${p}_r${start}_c${count}_${mode}"
archive="${stem}.tar.gz"
mkdir -p simulation_results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rc=0
if tar -xzf sepm_simulation_bundle.tar.gz; then
  python3 -u sim_batch.py --n-train "$n" --prevalence-percent "$p" \
    --rep-start "$start" --rep-count "$count" --mode "$mode" \
    --search-jobs 2 --output-root simulation_results || rc=$?
else
  rc=$?
fi
printf '%s\n' "$rc" > job_exit_status.txt
# Always package partial files and error status even when Python exits nonzero.
tar -czf "$archive" simulation_results job_exit_status.txt || exit 91
exit "$rc"
