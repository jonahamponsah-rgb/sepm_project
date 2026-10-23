#!/bin/bash
set -Eeuo pipefail

dataset="${1:?dataset required}"
fold="${2:?fold required}"
mode="${3:?pilot or benchmark required}"
archive="sepm_tmc_${dataset}_fold_${fold}_${mode}.tar.gz"

# Create the archive even when Python fails, so the error diagnostics return.
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > job_exit_status.txt
    mkdir -p results
    tar -czf "$archive" results job_exit_status.txt
    exit "$status"
}
trap finish EXIT

echo "Job started $(date -Is) on $(hostname)"
echo "Dataset=$dataset fold=$fold mode=$mode"
mkdir -p results
tar -xzf sepm_tmc_bundle.tar.gz
python -u scripts/chtc_fold_runner.py \
  --dataset "$dataset" --fold "$fold" --mode "$mode" \
  --data-dir data --output-root results --search-jobs 2

echo "Job finished $(date -Is)"
