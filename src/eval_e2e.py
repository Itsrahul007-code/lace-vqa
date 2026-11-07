"""End-to-end evaluation from raw images: our trained heads (frozen CLIP + head), and the
zero-shot ViLT / BLIP baselines. Supports image corruptions (robustness study), CPU, and
INT8 dynamic quantisation.

Examples:
  # our model on clean VizWiz val
  python src/eval_e2e.py --model lace --run_dir runs/vizwiz_lace_s0 --dataset vizwiz \
      --ann data/vizwiz/Annotations/val.json --image_dir data/vizwiz/val --out results/e2e_lace_clean.json
  # ViLT zero-shot under motion blur severity 3
  python src/eval_e2e.py --model vilt --dataset vizwiz --ann ... --image_dir ... \
      --corruption motion_blur --severity 3 --out results/e2e_vilt_motion3.json
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import corruptions
from backbone import ClipBackbone, load_rgb
from data import load_vizwiz, load_vqav2
from metrics import evaluate_predictions
from model import EndToEnd, LACEConfig, build_head

HF_DEFAULTS = {"vilt": "dandelin/vilt-b32-finetuned-vqa", "blip": "Salesforce/blip-vqa-base"}


class RawDataset(Dataset):
    def __init__(self, recs, image_dir, corruption, severity, max_side=0):
        self.recs, self.image_dir, self.c, self.s = recs, image_dir, corruption, severity
        self.max_side = max_side

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = load_rgb(os.path.join(self.image_dir, r["image"]))
        if self.max_side and max(img.size) > self.max_side:  # downscale first: faster, models resize anyway
            sc = self.max_side / max(img.size)
            img = img.resize((max(1, round(img.width * sc)), max(1, round(img.height * sc))), Image.BICUBIC)
        img = corruptions.apply(img, self.c, self.s, seed=i)
        return img, r["question"], i


def collate(b):
    imgs, qs, idx = zip(*b)
    return list(imgs), list(qs), list(idx)


def maybe_int8(model, int8, skip_vision=False):
    """Dynamic INT8 quantisation of Linear layers. With skip_vision (our models only), the CLIP vision
    encoder stays in FP32 and only the text encoder, the projections and the LACE head are quantised."""
    if not int8:
        return model
    from torch.ao.quantization import default_dynamic_qconfig, quantize_dynamic
    if skip_vision and hasattr(model, "backbone"):
        keep = ("backbone.clip.text_model", "backbone.clip.text_projection", "head")
        spec = {n: default_dynamic_qconfig for n, m in model.named_modules()
                if type(m) is torch.nn.Linear and n.startswith(keep)}
        return quantize_dynamic(model, qconfig_spec=spec, dtype=torch.qint8)
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class OursRunner:
    def __init__(self, run_dir, device, int8, int8_novision=False):
        ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
        self.vocab = ck["vocab"]
        cfg = LACEConfig(**ck["config"])
        head = build_head(ck["kind"], cfg)
        head.load_state_dict(ck["model"])
        bb = ClipBackbone(ck["backbone"], pool_grid=int(round(cfg.n_img_tokens ** 0.5)))
        self.bb, self.max_q_len = bb, cfg.max_q_len
        res_path = os.path.join(run_dir, "results.json")
        self.thr = json.load(open(res_path)).get("threshold") if os.path.exists(res_path) else None
        self.model = maybe_int8(EndToEnd(bb, head).eval(), int8, int8_novision).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, imgs, qs):
        px = torch.stack([self.bb.transform(im) for im in imgs]).to(self.device)
        enc = self.bb.tokenize(qs, self.max_q_len, self.device)
        o = self.model(px, enc["input_ids"], enc["attention_mask"])
        best = o["answer_logits"].argmax(1).tolist()
        p_ans = torch.sigmoid(o["answerable_logit"]).cpu().numpy()
        answers = [self.vocab[i] for i in best]
        if self.thr is not None and "unanswerable" in self.vocab:
            answers = ["unanswerable" if p < self.thr else a for a, p in zip(answers, p_ans)]
        return answers, p_ans


class ViltRunner:
    def __init__(self, name, device, int8):
        from transformers import ViltForQuestionAnswering, ViltProcessor
        self.proc = ViltProcessor.from_pretrained(name)
        self.model = maybe_int8(ViltForQuestionAnswering.from_pretrained(name).eval(), int8).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, imgs, qs):
        enc = self.proc(images=imgs, text=qs, return_tensors="pt", padding=True, truncation=True, max_length=40)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        id2label = self.model.config.id2label
        return [id2label[i] for i in logits.argmax(-1).tolist()], None


class BlipRunner:
    def __init__(self, name, device, int8):
        from transformers import BlipForQuestionAnswering, BlipProcessor
        self.proc = BlipProcessor.from_pretrained(name)
        self.model = maybe_int8(BlipForQuestionAnswering.from_pretrained(name).eval(), int8).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, imgs, qs):
        enc = self.proc(images=imgs, text=qs, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model.generate(**enc, max_new_tokens=10)
        return [s.strip() for s in self.proc.batch_decode(out, skip_special_tokens=True)], None


def build_runner(a, device):
    if a.model in ("lace", "clip_mlp"):
        return OursRunner(a.run_dir, device, a.int8, a.int8_novision)
    name = a.hf_model or HF_DEFAULTS[a.model]
    return (ViltRunner if a.model == "vilt" else BlipRunner)(name, device, a.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lace", "clip_mlp", "vilt", "blip"], required=True)
    ap.add_argument("--run_dir", help="training run dir (for lace / clip_mlp)")
    ap.add_argument("--hf_model", default=None, help="override HF id / local path for vilt or blip")
    ap.add_argument("--dataset", choices=["vizwiz", "vqav2"], required=True)
    ap.add_argument("--ann", help="VizWiz json")
    ap.add_argument("--q_json"); ap.add_argument("--a_json")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--corruption", default="none"); ap.add_argument("--severity", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_side", type=int, default=0, help="downscale images to this longest side before corruption")
    ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--int8", action="store_true", help="dynamic INT8 quantisation (CPU only)")
    ap.add_argument("--int8_novision", action="store_true",
                    help="with --int8: keep the CLIP vision encoder in FP32 (our models only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.int8_novision:
        a.int8 = True
    if a.int8:
        a.device = "cpu"
    device = torch.device(a.device)

    recs = load_vizwiz(a.ann) if a.dataset == "vizwiz" else load_vqav2(a.q_json, a.a_json, "COCO_val2014_")
    recs = [r for r in recs if r["answers"]]
    if a.max_samples and a.max_samples < len(recs):
        recs = random.Random(0).sample(recs, a.max_samples)
    runner = build_runner(a, device)
    dl = DataLoader(RawDataset(recs, a.image_dir, a.corruption, a.severity, a.max_side), batch_size=a.batch_size,
                    num_workers=a.workers, collate_fn=collate)
    answers, p_ans = [None] * len(recs), np.full(len(recs), np.nan)
    t0 = time.time()
    for imgs, qs, idx in dl:
        ans, p = runner(imgs, qs)
        for j, i in enumerate(idx):
            answers[i] = ans[j]
            if p is not None:
                p_ans[i] = p[j]
    elapsed = time.time() - t0
    res = evaluate_predictions(answers, recs, None if np.isnan(p_ans).all() else p_ans)
    res.update({"model": a.model, "run_dir": a.run_dir, "dataset": a.dataset, "corruption": a.corruption,
                "severity": a.severity, "max_side": a.max_side, "int8": a.int8, "int8_novision": a.int8_novision, "device": a.device, "n_samples": len(recs),
                "seconds": elapsed, "throughput_qps": len(recs) / max(elapsed, 1e-9)})
    if not np.isnan(p_ans).all():
        res["mean_p_answerable"] = float(np.nanmean(p_ans))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("model", "corruption", "severity", "accuracy", "n")}))


if __name__ == "__main__":
    main()
