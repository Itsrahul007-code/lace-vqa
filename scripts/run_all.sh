#!/usr/bin/env bash
# Full experimental pipeline. Every stage is resumable (finished outputs are skipped).
#   STAGES="feat_viz train_main" bash scripts/run_all.sh
# Stages: feat_viz train_main ablations backbones zeroshot robustness int8acc benchmark
#         feat_vqa vqa_pretrain tables
set -euo pipefail
export TOKENIZERS_PARALLELISM=false   # avoids rare DataLoader-fork deadlocks with HF fast tokenizers
cd "$(dirname "$0")/.."

DATA=${DATA:-data}          # raw datasets
WORK=${WORK:-work}          # features / runs / results (put this on Google Drive to survive disconnects)
SEEDS=${SEEDS:-"0 1 2"}
BACKBONE=${BACKBONE:-openai/clip-vit-base-patch16}     # main model backbone
# backbone-size study: short_name=hf_id
BACKBONES=${BACKBONES:-"tiny8m=wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M tiny40m=wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M b32=openai/clip-vit-base-patch32"}
VILT_MODEL=${VILT_MODEL:-dandelin/vilt-b32-finetuned-vqa}
BLIP_MODEL=${BLIP_MODEL:-Salesforce/blip-vqa-base}
EPOCHS=${EPOCHS:-15}
LACE_ARGS=${LACE_ARGS:---residual_fusion --n_layers 1 --lr 3e-4 --lr_global 1e-3}   # final LACE (v2) settings; extra args added after it override them
ROB_N=${ROB_N:-1000}        # questions per robustness setting
ROB_SEV=${ROB_SEV:-"1 3 5"}
ROB_RUNS=${ROB_RUNS:-"viz_lace viz_clipmlp viz_lace_tiny8m"}
PAPER=${PAPER:-paper}     # where results_macros.tex and figures are written
WORKERS=${WORKERS:-2}       # Colab/Kaggle have 2-4 CPU cores
STAGES=${STAGES:-"feat_viz train_main ablations backbones zeroshot robustness int8acc benchmark tables"}

VZ=$DATA/vizwiz
FEAT=$WORK/feats; RUNS=$WORK/runs; RES=$WORK/results
mkdir -p "$FEAT" "$RUNS" "$RES"
PY="python src"
has () { [[ " $STAGES " == *" $1 "* ]]; }
hf () { [ "$1" = vilt ] && echo "$VILT_MODEL" || echo "$BLIP_MODEL"; }

viz_args () {  # feature-suffix backbone
  echo "--dataset vizwiz --train_json $VZ/Annotations/train.json --test_json $VZ/Annotations/val.json \
        --feat_train $FEAT/vizwiz_train$1 --feat_test $FEAT/vizwiz_val$1 --backbone $2 --workers $WORKERS"
}
VIZ_ARGS=$(viz_args "" "$BACKBONE")
VQA_ARGS="--dataset vqav2 --train_q $DATA/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
          --train_a $DATA/vqav2/v2_mscoco_train2014_annotations.json \
          --test_q $DATA/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
          --test_a $DATA/vqav2/v2_mscoco_val2014_annotations.json \
          --feat_train $FEAT/coco_train2014 --feat_test $FEAT/coco_val2014 --backbone $BACKBONE --workers $WORKERS"
E2E="--dataset vizwiz --ann $VZ/Annotations/val.json --image_dir $VZ/val --workers $WORKERS"

train () {  # name seed extra-args...
  local name=$1 seed=$2; shift 2
  if [ -f "$RUNS/${name}_s${seed}/results.json" ]; then echo "skip $name s$seed"; return; fi
  $PY/train.py --epochs "$EPOCHS" "$@" --seed "$seed" --out_dir "$RUNS/${name}_s${seed}"
}
feats () {  # image_dir out_prefix backbone
  [ -f "$2_index.json" ] || $PY/features.py --image_dir "$1" --out "$2" --backbone "$3" --workers $WORKERS
}
e2e () {  # out_file args...
  local f=$1; shift
  [ -f "$f" ] || $PY/eval_e2e.py "$@" --out "$f"
}

# ---------------------------------------------------------------- 1. features (VizWiz)
if has feat_viz; then
  feats "$VZ/train" "$FEAT/vizwiz_train" "$BACKBONE"
  feats "$VZ/val"   "$FEAT/vizwiz_val"   "$BACKBONE"
fi

# ---------------------------------------------------------------- 2. main VizWiz results
if has train_main; then
  for s in $SEEDS; do
    train viz_clipmlp $s $VIZ_ARGS --model clip_mlp --lr 1e-3
    train viz_lace    $s $VIZ_ARGS --model lace $LACE_ARGS
  done
fi

# ---------------------------------------------------------------- 3. ablations (VizWiz)
if has ablations; then
  for s in $SEEDS; do
    train viz_lace_L0       $s $VIZ_ARGS --model lace $LACE_ARGS --n_layers 0
    train viz_lace_L2       $s $VIZ_ARGS --model lace $LACE_ARGS --n_layers 2
    train viz_lace_L4       $s $VIZ_ARGS --model lace $LACE_ARGS --n_layers 4
    train viz_lace_uni      $s $VIZ_ARGS --model lace $LACE_ARGS --no_bidirectional
    train viz_lace_noglobal $s $VIZ_ARGS --model lace $LACE_ARGS --no_global
    train viz_lace_nopatch  $s $VIZ_ARGS --model lace $LACE_ARGS --no_patch
    train viz_lace_noaux    $s $VIZ_ARGS --model lace $LACE_ARGS --w_answerable 0 --w_type 0
    train viz_lace_k1000    $s $VIZ_ARGS --model lace $LACE_ARGS --top_k_answers 1000
    train viz_lace_d256     $s $VIZ_ARGS --model lace $LACE_ARGS --d_model 256 --n_heads 4
    train viz_lace_frac25   $s $VIZ_ARGS --model lace $LACE_ARGS --max_train 4600
    train viz_lace_frac50   $s $VIZ_ARGS --model lace $LACE_ARGS --max_train 9200
  done
fi

# ---------------------------------------------------------------- 4. backbone-size study
if has backbones; then
  for pair in $BACKBONES; do
    short=${pair%%=*}; id=${pair#*=}
    feats "$VZ/train" "$FEAT/vizwiz_train_$short" "$id"
    feats "$VZ/val"   "$FEAT/vizwiz_val_$short"   "$id"
    for s in $SEEDS; do
      train "viz_lace_$short" $s $(viz_args "_$short" "$id") --model lace $LACE_ARGS
    done
  done
fi

# ---------------------------------------------------------------- 5. zero-shot baselines, clean VizWiz val
if has zeroshot; then
  for m in vilt blip; do
    e2e "$RES/zs_${m}.json" --model $m --hf_model "$(hf $m)" $E2E --workers 0 --batch_size 32
  done
fi

# ---------------------------------------------------------------- 6. robustness to BLV-style degradations
if has robustness; then
  for r in $ROB_RUNS; do   # clean reference on the same subset
    e2e "$RES/rob_${r}__none_0.json" --model lace --run_dir "$RUNS/${r}_s0" $E2E --max_samples $ROB_N --max_side 640
  done
  for m in vilt blip; do
    e2e "$RES/rob_${m}__none_0.json" --model $m --hf_model "$(hf $m)" $E2E --workers 0 --max_samples $ROB_N --max_side 640
  done
  for c in gaussian_blur motion_blur low_light overexposure gaussian_noise jpeg poor_framing rotation; do
    for sev in $ROB_SEV; do
      for r in $ROB_RUNS; do
        e2e "$RES/rob_${r}__${c}_${sev}.json" --model lace --run_dir "$RUNS/${r}_s0" $E2E \
            --max_samples $ROB_N --max_side 640 --corruption $c --severity $sev
      done
      for m in vilt blip; do
        e2e "$RES/rob_${m}__${c}_${sev}.json" --model $m --hf_model "$(hf $m)" $E2E --workers 0 \
            --max_samples $ROB_N --max_side 640 --corruption $c --severity $sev
      done
    done
  done
fi

# ---------------------------------------------------------------- 7. INT8 accuracy (CPU, subset)
if has int8acc; then
  for r in viz_lace viz_lace_tiny8m; do
    [ -d "$RUNS/${r}_s0" ] || continue
    e2e "$RES/int8_${r}.json"    --model lace --run_dir "$RUNS/${r}_s0" $E2E --max_samples $ROB_N --int8
    e2e "$RES/int8nv_${r}.json"  --model lace --run_dir "$RUNS/${r}_s0" $E2E --max_samples $ROB_N --int8_novision
    e2e "$RES/fp32cpu_${r}.json" --model lace --run_dir "$RUNS/${r}_s0" $E2E --max_samples $ROB_N --device cpu
  done
fi

# ---------------------------------------------------------------- 8. efficiency
if has benchmark; then
  OURS=""
  for r in viz_lace viz_clipmlp viz_lace_tiny8m viz_lace_tiny40m viz_lace_b32 viz_lace_d256; do
    [ -f "$RUNS/${r}_s0/best.pt" ] && OURS="$OURS ${r}=$RUNS/${r}_s0"
  done
  [ -f "$RES/benchmark.json" ] || $PY/benchmark.py --ours $OURS --baselines vilt blip \
     --vilt_model "$VILT_MODEL" --blip_model "$BLIP_MODEL" --out "$RES/benchmark.json" ${BENCH_ARGS:-}
fi

# ---------------------------------------------------------------- 9. optional: VQA v2 pre-training + transfer
if has feat_vqa; then
  feats "$DATA/coco/train2014" "$FEAT/coco_train2014" "$BACKBONE"
  feats "$DATA/coco/val2014"   "$FEAT/coco_val2014"   "$BACKBONE"
fi
if has vqa_pretrain; then
  train vqa_clipmlp 0 $VQA_ARGS --model clip_mlp --lr 1e-3 --epochs 10
  train vqa_lace    0 $VQA_ARGS --model lace $LACE_ARGS --epochs 10
  for s in $SEEDS; do
    train viz_lace_vqapt $s $VIZ_ARGS --model lace $LACE_ARGS --init_from "$RUNS/vqa_lace_s0/best.pt" --lr 1e-4
  done
fi

# ---------------------------------------------------------------- 10. tables + figures for the paper
if has tables; then
  $PY/make_tables.py --runs "$RUNS" --e2e "$RES" --bench "$RES/benchmark.json" --paper "$PAPER"
fi
