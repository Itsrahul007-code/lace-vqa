#!/usr/bin/env bash
# Download VizWiz-VQA (required) and VQA v2 (optional) into $DATA.
# Usage: DATA=/content/data bash scripts/download_data.sh vizwiz [vizwiz_val] [vqav2]
#   vizwiz      annotations + train + val images
#   vizwiz_val  annotations + val images only (enough for zero-shot / robustness / INT8 stages)
# Optional ZIP_CACHE (e.g. a Google Drive folder): every zip is downloaded only once and kept there;
# later sessions copy it from the cache instead of downloading again.
# Local zips are deleted after extraction to save disk.
set -euo pipefail
DATA=${DATA:-data}
ZIP_CACHE=${ZIP_CACHE:-}
mkdir -p "$DATA"
[ -n "$ZIP_CACHE" ] && mkdir -p "$ZIP_CACHE"

fetch () {  # url dest_dir
  local url=$1 dir=$2 name f
  name=$(basename "$url"); f="$dir/$name"
  mkdir -p "$dir"
  if [ -f "$f.done" ]; then return; fi
  if [ -n "$ZIP_CACHE" ] && [ -f "$ZIP_CACHE/$name" ]; then
    echo ">> cache: $ZIP_CACHE/$name"
    cp "$ZIP_CACHE/$name" "$f"
  else
    echo ">> $url"
    wget -c -q --show-progress -O "$f" "$url" || {
      echo "!! Download failed: $url"
      echo "   Get the file manually from the dataset website and place it in ${ZIP_CACHE:-$dir}"; exit 1; }
    if [ -n "$ZIP_CACHE" ]; then cp "$f" "$ZIP_CACHE/$name.part" && mv "$ZIP_CACHE/$name.part" "$ZIP_CACHE/$name"; fi
  fi
  unzip -q -o "$f" -d "$dir" && rm -f "$f" && touch "$f.done"
}

fix_ann () {  # Annotations.zip unpacks the json files flat; move them into Annotations/
  mkdir -p "$DATA/vizwiz/Annotations"
  for j in train val test; do
    if [ -f "$DATA/vizwiz/$j.json" ]; then mv -f "$DATA/vizwiz/$j.json" "$DATA/vizwiz/Annotations/"; fi
  done
  ls "$DATA/vizwiz/Annotations"
}

for ds in "$@"; do
  case $ds in
    vizwiz)
      # Official VizWiz-VQA files (https://vizwiz.org/tasks-and-datasets/vqa/).
      # If these links change, download train/val/test images and Annotations from that page.
      B=https://vizwiz.cs.colorado.edu/VizWiz_final
      fetch $B/vqa_data/Annotations.zip "$DATA/vizwiz"
      fetch $B/images/train.zip "$DATA/vizwiz"
      fetch $B/images/val.zip "$DATA/vizwiz"
      # test images are only needed for an EvalAI leaderboard submission:
      # fetch $B/images/test.zip "$DATA/vizwiz"
      fix_ann
      ls "$DATA/vizwiz"
      ;;
    vizwiz_val)
      B=https://vizwiz.cs.colorado.edu/VizWiz_final
      fetch $B/vqa_data/Annotations.zip "$DATA/vizwiz"
      fetch $B/images/val.zip "$DATA/vizwiz"
      fix_ann
      ls "$DATA/vizwiz"
      ;;
    vqav2)
      S=https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa
      fetch $S/v2_Questions_Train_mscoco.zip "$DATA/vqav2"
      fetch $S/v2_Annotations_Train_mscoco.zip "$DATA/vqav2"
      fetch $S/v2_Questions_Val_mscoco.zip "$DATA/vqav2"
      fetch $S/v2_Annotations_Val_mscoco.zip "$DATA/vqav2"
      fetch http://images.cocodataset.org/zips/train2014.zip "$DATA/coco"
      fetch http://images.cocodataset.org/zips/val2014.zip "$DATA/coco"
      ls "$DATA/vqav2" "$DATA/coco"
      ;;
    *) echo "unknown dataset $ds"; exit 1;;
  esac
done
