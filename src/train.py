"""Train and evaluate LACE-VQA (or the CLIP-MLP baseline) on cached CLIP features.

VizWiz example:
  python src/train.py --dataset vizwiz --model lace \
     --train_json data/vizwiz/Annotations/train.json --test_json data/vizwiz/Annotations/val.json \
     --feat_train feats/vizwiz_train --feat_test feats/vizwiz_val --out_dir runs/vizwiz_lace_s0 --seed 0
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone import ClipBackbone
from data import (ANSWER_TYPES, FeatureStore, VQAFeatureDataset, build_vocab, load_vizwiz, load_vqav2,
                  split_by_image, vocab_coverage)
from metrics import evaluate_predictions
from model import LACEConfig, build_head, count_params, vqa_loss


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_split(args, which):
    """which in {train, test}. Returns list of records."""
    if args.dataset == "vizwiz":
        path = args.train_json if which == "train" else args.test_json
        return load_vizwiz(path)
    if which == "train":
        return load_vqav2(args.train_q, args.train_a, "COCO_train2014_")
    return load_vqav2(args.test_q, args.test_a, "COCO_val2014_")


@torch.no_grad()
def predict(head, backbone, loader, device, amp):
    head.eval()
    outs = {"answer_logits": [], "answerable_prob": [], "type_prob": [], "idx": []}
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            txt, txt_g = backbone.encode_text(b["input_ids"], b["attention_mask"])
            o = head(b["patch"], b["img_global"], txt.float(), b["attention_mask"], txt_g.float())
        outs["answer_logits"].append(o["answer_logits"].float().cpu())
        outs["answerable_prob"].append(torch.sigmoid(o["answerable_logit"].float()).cpu())
        outs["type_prob"].append(torch.softmax(o["type_logits"].float(), -1).cpu())
        outs["idx"].append(b["idx"].cpu())
    return {k: torch.cat(v).numpy() for k, v in outs.items()}


def decode(pred, vocab, threshold=None):
    """Arg-max answer; if threshold is given and 'unanswerable' is in the vocabulary, answer
    'unanswerable' whenever P(answerable) < threshold (answerability gate)."""
    best = pred["answer_logits"].argmax(1)
    answers = [vocab[i] for i in best]
    if threshold is not None and "unanswerable" in vocab:
        answers = ["unanswerable" if p < threshold else a for a, p in zip(answers, pred["answerable_prob"])]
    return answers


def tune_threshold(pred, recs, vocab):
    best_t, best_acc = None, evaluate_predictions(decode(pred, vocab), recs)["accuracy"]
    if "unanswerable" not in vocab:
        return None, best_acc
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        acc = evaluate_predictions(decode(pred, vocab, t), recs)["accuracy"]
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vizwiz", "vqav2"], required=True)
    ap.add_argument("--model", choices=["lace", "clip_mlp"], default="lace")
    # annotations
    ap.add_argument("--train_json"); ap.add_argument("--test_json")
    ap.add_argument("--train_q"); ap.add_argument("--train_a"); ap.add_argument("--test_q"); ap.add_argument("--test_a")
    # cached features (prefixes written by features.py)
    ap.add_argument("--feat_train", required=True); ap.add_argument("--feat_test", required=True)
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out_dir", required=True)
    # model
    ap.add_argument("--d_model", type=int, default=384); ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--residual_fusion", action="store_true", help="v2: wide global MLP + zero-init token residual")
    ap.add_argument("--global_hidden", type=int, default=1024); ap.add_argument("--global_dropout", type=float, default=0.5)
    ap.add_argument("--global_inputs", type=int, default=4, choices=[2, 4])
    ap.add_argument("--lr_global", type=float, default=0.0,
                    help="separate learning rate for the global branch and output heads (0 = same as --lr)")
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--no_bidirectional", action="store_true")
    ap.add_argument("--no_patch", action="store_true"); ap.add_argument("--no_global", action="store_true")
    ap.add_argument("--top_k_answers", type=int, default=3000)
    ap.add_argument("--max_q_len", type=int, default=32)
    # loss
    ap.add_argument("--w_answerable", type=float, default=0.5); ap.add_argument("--w_type", type=float, default=0.2)
    # optimisation
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--max_train", type=int, default=0, help="subsample training questions (0 = all)")
    ap.add_argument("--init_from", default=None, help="checkpoint to initialise from (e.g. VQA v2 pretraining)")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.residual_fusion and args.no_global:
        print("note: --no_global disables residual fusion (token path only)")
        args.residual_fusion = False

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    amp = device.type == "cuda"

    # ---------------- data ----------------
    all_train = load_split(args, "train")
    train_recs, dev_recs = split_by_image(all_train, args.dev_frac, seed=0)  # fixed dev split across seeds
    if args.max_train and args.max_train < len(train_recs):
        train_recs = random.Random(args.seed).sample(train_recs, args.max_train)
    test_recs = load_split(args, "test")
    vocab = build_vocab(train_recs, args.top_k_answers)
    cov = {"dev": vocab_coverage(dev_recs, vocab), "test": vocab_coverage(test_recs, vocab)}
    print(f"train {len(train_recs)} | dev {len(dev_recs)} | test {len(test_recs)} | vocab {len(vocab)} "
          f"| oracle acc dev {cov['dev']:.4f} test {cov['test']:.4f}")
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    backbone = ClipBackbone(args.backbone)
    backbone.to(device)
    if amp:
        backbone.half()
    st_train, st_test = FeatureStore(args.feat_train), FeatureStore(args.feat_test)
    mk = lambda recs, stores: VQAFeatureDataset(recs, stores, vocab, backbone.tokenizer, args.max_q_len)
    ds_train, ds_dev, ds_test = mk(train_recs, [st_train]), mk(dev_recs, [st_train]), mk(test_recs, [st_test])
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
                                   pin_memory=amp, drop_last=sh and len(ds) >= 2 * args.batch_size,
                                   persistent_workers=args.workers > 0)
    dl_train, dl_dev, dl_test = dl(ds_train, True), dl(ds_dev, False), dl(ds_test, False)

    # ---------------- model ----------------
    cfg = LACEConfig(dim_vision=backbone.dim_vision, dim_text=backbone.dim_text, dim_proj=backbone.dim_proj,
                     n_img_tokens=st_train.patch.shape[1], max_q_len=args.max_q_len, d_model=args.d_model,
                     n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, bidirectional=not args.no_bidirectional,
                     residual_fusion=args.residual_fusion, global_hidden=args.global_hidden,
                     global_dropout=args.global_dropout, global_inputs=args.global_inputs,
                     use_patch=not args.no_patch, use_global=not args.no_global, n_answers=len(vocab),
                     n_types=len(ANSWER_TYPES))
    head = build_head(args.model, cfg).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        src_vocab = ck["vocab"]
        sd = ck["model"]
        own = head.state_dict()
        # copy answer-classifier rows for answers shared by both vocabularies
        for k in ["answer_head.weight", "answer_head.bias"]:
            if k in sd:
                new = own[k].clone()
                s2i = {a: i for i, a in enumerate(src_vocab)}
                for j, a in enumerate(vocab):
                    if a in s2i:
                        new[j] = sd[k][s2i[a]]
                sd[k] = new
        missing, unexpected = head.load_state_dict({k: v for k, v in sd.items() if k in own and
                                                    own[k].shape == v.shape}, strict=False)
        print(f"init_from {args.init_from}: {len(missing)} tensors kept at random init")
    n_params = count_params(head)
    print(f"trainable head parameters: {n_params/1e6:.2f}M")

    if args.lr_global > 0:
        fast = ("glob.", "answer_head.", "answerable_head.", "type_head.", "net.")
        g1 = [p for n, p in head.named_parameters() if n.startswith(fast)]
        g2 = [p for n, p in head.named_parameters() if not n.startswith(fast)]
        opt = torch.optim.AdamW([{"params": g1, "lr": args.lr_global}, {"params": g2, "lr": args.lr}],
                                weight_decay=args.weight_decay)
        print(f"param groups: {sum(p.numel() for p in g1)/1e6:.2f}M @ lr {args.lr_global}, "
              f"{sum(p.numel() for p in g2)/1e6:.2f}M @ lr {args.lr}")
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = max(1, args.epochs * len(dl_train))
    warm = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_acc, best_epoch = -1.0, -1
    log_f = open(os.path.join(args.out_dir, "log.jsonl"), "w")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        head.train()
        run = 0.0
        for b in tqdm(dl_train, desc=f"epoch {ep}", leave=False):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                txt, txt_g = backbone.encode_text(b["input_ids"], b["attention_mask"])
                out = head(b["patch"], b["img_global"], txt.float(), b["attention_mask"], txt_g.float())
            loss, logs = vqa_loss({k: v.float() for k, v in out.items()}, b, args.w_answerable, args.w_type)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        pred = predict(head, backbone, dl_dev, device, amp)
        dev_acc = evaluate_predictions(decode(pred, vocab), dev_recs)["accuracy"]
        rec = {"epoch": ep, "train_loss": run / len(dl_train), "dev_acc": dev_acc, "time_s": time.time() - t0}
        print(json.dumps(rec))
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if dev_acc > best_acc:
            best_acc, best_epoch = dev_acc, ep
            torch.save({"model": head.state_dict(), "config": cfg.to_dict(), "vocab": vocab, "kind": args.model,
                        "backbone": args.backbone}, os.path.join(args.out_dir, "best.pt"))
    log_f.close()

    # ---------------- final evaluation with the best checkpoint ----------------
    head.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pt"), map_location="cpu")["model"])
    pred_dev = predict(head, backbone, dl_dev, device, amp)
    thr, _ = tune_threshold(pred_dev, dev_recs, vocab)
    pred_test = predict(head, backbone, dl_test, device, amp)
    res = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed, "backbone": args.backbone,
        "config": cfg.to_dict(), "trainable_params": n_params, "best_epoch": best_epoch,
        "train_time_s": time.time() - t0, "n_train": len(train_recs), "max_train": args.max_train,
        "init_from": args.init_from, "w_answerable": args.w_answerable, "w_type": args.w_type,
        "vocab_size": len(vocab), "oracle_acc": cov, "threshold": thr,
        "dev": {"argmax": evaluate_predictions(decode(pred_dev, vocab), dev_recs, pred_dev["answerable_prob"]),
                "gated": evaluate_predictions(decode(pred_dev, vocab, thr), dev_recs, pred_dev["answerable_prob"])},
        "test": {"argmax": evaluate_predictions(decode(pred_test, vocab), test_recs, pred_test["answerable_prob"]),
                 "gated": evaluate_predictions(decode(pred_test, vocab, thr), test_recs,
                                               pred_test["answerable_prob"])},
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    final = decode(pred_test, vocab, thr)
    with open(os.path.join(args.out_dir, "predictions_test.json"), "w") as f:
        json.dump([{"qid": r["qid"], "image": r["image"], "question": r["question"], "pred": a,
                    "p_answerable": float(p), "gt": r["answers"], "answer_type": r["answer_type"]}
                   for r, a, p in zip(test_recs, final, pred_test["answerable_prob"])], f)
    print("TEST:", json.dumps(res["test"]["gated"], indent=1))


if __name__ == "__main__":
    main()
