# LACE-VQA: lightweight, answerability-aware VQA for blind and low-vision users

Code, trained-model results and evaluation outputs for the paper
**"Lightweight Transformer Architectures for Real-Time Visual Question Answering in Assistive Multimedia Applications"**
(submitted to *Visual Computing for Industry, Biomedicine, and Art*).

LACE-VQA is a small head (8–13M trainable parameters) on a **frozen CLIP encoder**. A wide classifier on global CLIP
features gives the base prediction, and a question-guided co-attention read-out of image-patch and question tokens adds
a zero-initialised residual correction. The head predicts the answer, whether the question is **answerable** from the
photograph, and the answer type.

## Main results (VizWiz-VQA validation, 4,319 questions)

| Model | Trainable params | VQA acc. (%) | Yes/no (%) | Answerability AP (%) |
|---|---|---|---|---|
| ViLT, zero-shot | – | 17.73 | – | – |
| BLIP, zero-shot | – | 17.14 | – | – |
| CLIP-MLP (ViT-B/16, our baseline) | 4.13M | 63.70 ± 0.20 | 57.45 | 95.04 |
| **LACE-VQA (ViT-B/16)** | 12.52M | **64.03 ± 0.29** | **69.34** | 94.37 |
| LACE-VQA (TinyCLIP-8M) | 12.22M | 60.53 ± 0.24 | 66.75 | 93.29 |

Mean ± std over 5 seeds (main models) or 3 seeds (other encoders). The overall difference between LACE-VQA and
CLIP-MLP is **not statistically significant** (paired bootstrap, p = 0.14); the yes/no difference is (p < 0.001).

**Efficiency** (full image + question → answer, batch 1, one CPU thread of a Colab VM): LACE-TinyCLIP INT8 66 ms,
LACE-B/16 FP32 482 ms, ViLT 467 ms, BLIP 1,749 ms. On a T4 GPU: LACE-B/16 33 ms.

**Robustness** (8 synthetic degradations × 3 severities, 1,000 questions): accuracy retained on average —
LACE-B/16 94.4%, CLIP-MLP 95.3%, LACE-Tiny 95.9%.

All numbers come from the files in `results/`; `results/summary_macros.json` contains every value used in the paper.

## Repository layout

```
src/            model, training, evaluation, benchmark, corruptions, table generation
scripts/        download_data.sh (VizWiz), run_all.sh (all experiments, resumable stages)
notebooks/      run_all_colab.ipynb (full pipeline), resume_eval_colab.ipynb (evaluation stages only)
tests/          offline smoke test on synthetic data with tiny models
results/runs/   results.json + training log for every trained model and seed
results/eval/   zero-shot, robustness (rob_*), INT8 and benchmark outputs
figures/        robustness and training-curve plots
```

## Reproduce

Everything ran on a free Google Colab T4 GPU.

```bash
pip install -r requirements.txt
bash tests/smoke_test.sh                        # optional: checks the code runs (CPU, synthetic data)
DATA=data bash scripts/download_data.sh vizwiz  # VizWiz-VQA images + annotations
STAGES="feat_viz train_main ablations backbones zeroshot robustness int8acc benchmark tables" \
  SEEDS="0 1 2" DATA=data WORK=work bash scripts/run_all.sh
```

For the 5-seed main results use `SEEDS="0 1 2 3 4" STAGES="train_main"`. In Colab, open
`notebooks/run_all_colab.ipynb` (Runtime → T4 GPU) and run the cells in order; each stage is resumable.

Data: VizWiz-VQA, <https://vizwiz.org/tasks-and-datasets/vqa/> (not redistributed here; see its licence).
Pre-trained encoders are downloaded from Hugging Face: `openai/clip-vit-base-patch16`, `openai/clip-vit-base-patch32`,
`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`, `wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M`,
`dandelin/vilt-b32-finetuned-vqa`, `Salesforce/blip-vqa-base`.

## Trained heads

Trained heads (`best.pt`, containing weights, configuration and answer vocabulary) are attached to the GitHub
release of this repository. Evaluate one on raw images with:

```bash
python src/eval_e2e.py --model lace --run_dir <folder containing best.pt and results.json> --dataset vizwiz \
  --ann data/vizwiz/Annotations/val.json --image_dir data/vizwiz/val --out out.json
```

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## Licence

Code: MIT (see `LICENSE`). Datasets and pre-trained models keep their own licences.
