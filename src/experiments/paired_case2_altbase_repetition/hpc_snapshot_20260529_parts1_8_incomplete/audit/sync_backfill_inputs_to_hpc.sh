#!/bin/bash
set -euo pipefail

REMOTE='e0829076@atlas9.nus.edu.sg'
REMOTE_PROJECT_ROOT=$HOME/2026-tensor
REMOTE_MANIFEST_DIR=$HOME/2026-tensor/hpc/manifests/case2_backfill_20260529

scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/audit_case2_hpc_parts.py' "$REMOTE:$REMOTE_PROJECT_ROOT/src/experiments/paired_case2_altbase_repetition/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/paired_case2_altbase_backfill.py' "$REMOTE:$REMOTE_PROJECT_ROOT/src/experiments/paired_case2_altbase_repetition/"
scp '/Users/lin/Desktop/Research/2026-tensor/hpc/paired_case2_altbase_backfill_parallel.pbs' "$REMOTE:$REMOTE_PROJECT_ROOT/hpc/"
ssh "$REMOTE" "mkdir -p $REMOTE_MANIFEST_DIR"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part1_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part2_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part3_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part4_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part5_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part6_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part7_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"
scp '/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/hpc_snapshot_20260529_parts1_8_incomplete/audit/part8_missing.csv' "$REMOTE:$REMOTE_MANIFEST_DIR/"

