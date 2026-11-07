#!/usr/bin/env bash
# Offline end-to-end test on a tiny synthetic dataset with tiny random models (~5-10 min on CPU).
# It checks that every stage runs; the numbers it produces are meaningless.
set -euo pipefail
cd "$(dirname "$0")/.."
S=${SMOKE_DIR:-/tmp/access_vqa_smoke}
rm -rf "$S"
python tests/make_synthetic.py --out "$S"
cp -r paper "$S/paper"
DATA=$S/data WORK=$S/work PAPER=$S/paper SEEDS="0 1" EPOCHS=2 WORKERS=0 \
BACKBONE=$S/models/tiny-clip BACKBONES="tiny8m=$S/models/tiny-clip" \
VILT_MODEL=$S/models/tiny-vilt BLIP_MODEL=$S/models/tiny-blip \
ROB_N=40 ROB_SEV="5" BENCH_ARGS="--warmup 1 --iters 3 --no_gpu" \
STAGES="feat_viz train_main ablations backbones zeroshot robustness int8acc benchmark feat_vqa vqa_pretrain tables" \
  bash scripts/run_all.sh
echo "SMOKE TEST PASSED -- outputs in $S (do not use these numbers)"
