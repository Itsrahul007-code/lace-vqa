"""Dataset loading for VizWiz-VQA and VQA v2, answer vocabulary, and cached-feature datasets."""
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from vqa_utils import normalize_answer, soft_scores

ANSWER_TYPES = ["yes/no", "number", "other", "unanswerable"]
TYPE2ID = {t: i for i, t in enumerate(ANSWER_TYPES)}


# --------------------------------------------------------------------------------------
# Annotation loaders -> list of records
#   record = {qid, image, question, answers (list[str] normalised, may be empty for test),
#             answer_type (str|None), answerable (0/1/None)}
# --------------------------------------------------------------------------------------
def load_vizwiz(json_path: str) -> List[Dict]:
    with open(json_path) as f:
        items = json.load(f)
    recs = []
    for i, it in enumerate(items):
        answers = [normalize_answer(a["answer"]) for a in it.get("answers", [])]
        recs.append({
            "qid": it.get("question_id", it["image"]),
            "image": it["image"],
            "question": it["question"],
            "answers": answers,
            "answer_type": it.get("answer_type"),
            "answerable": it.get("answerable"),
        })
    return recs


def load_vqav2(questions_json: str, annotations_json: Optional[str], image_prefix: str) -> List[Dict]:
    """image_prefix e.g. 'COCO_train2014_' -> file COCO_train2014_000000123456.jpg"""
    with open(questions_json) as f:
        qs = json.load(f)["questions"]
    anns = {}
    if annotations_json and os.path.exists(annotations_json):
        with open(annotations_json) as f:
            anns = {a["question_id"]: a for a in json.load(f)["annotations"]}
    recs = []
    for q in qs:
        a = anns.get(q["question_id"])
        recs.append({
            "qid": q["question_id"],
            "image": f"{image_prefix}{q['image_id']:012d}.jpg",
            "question": q["question"],
            "answers": [normalize_answer(x["answer"]) for x in a["answers"]] if a else [],
            "answer_type": a["answer_type"] if a else None,
            "answerable": None,  # VQA v2 has no answerability labels
        })
    return recs


def split_by_image(recs: List[Dict], dev_frac: float, seed: int = 0):
    """Hold out a dev split from training data, grouped by image (no image leakage)."""
    images = sorted({r["image"] for r in recs})
    rng = random.Random(seed)
    rng.shuffle(images)
    dev_imgs = set(images[: int(len(images) * dev_frac)])
    train = [r for r in recs if r["image"] not in dev_imgs]
    dev = [r for r in recs if r["image"] in dev_imgs]
    return train, dev


# --------------------------------------------------------------------------------------
# Answer vocabulary
# --------------------------------------------------------------------------------------
def build_vocab(recs: List[Dict], top_k: int, min_count: int = 1) -> List[str]:
    cnt = Counter()
    for r in recs:
        # count each distinct answer once per question (as in common VQA practice)
        cnt.update(set(r["answers"]))
    vocab = [a for a, c in cnt.most_common() if c >= min_count and a != ""][:top_k]
    return vocab


def vocab_coverage(recs: List[Dict], vocab: List[str]) -> float:
    """Upper bound: mean VQA accuracy achievable if the best in-vocab answer is always predicted."""
    vs = set(vocab)
    tot = 0.0
    n = 0
    for r in recs:
        if not r["answers"]:
            continue
        sc = soft_scores(r["answers"])
        tot += max([s for a, s in sc.items() if a in vs], default=0.0)
        n += 1
    return tot / max(n, 1)


# --------------------------------------------------------------------------------------
# Cached-feature dataset
# --------------------------------------------------------------------------------------
class FeatureStore:
    """Memory-mapped image features written by features.py."""

    def __init__(self, prefix: str):
        self.patch = np.load(prefix + "_patch.npy", mmap_mode="r")
        self.glob = np.load(prefix + "_global.npy", mmap_mode="r")
        with open(prefix + "_index.json") as f:
            self.index = json.load(f)

    def row(self, image: str) -> int:
        return self.index[image]


class VQAFeatureDataset(Dataset):
    def __init__(self, recs: List[Dict], stores: List[FeatureStore], vocab: List[str],
                 tokenizer, max_q_len: int = 32):
        self.recs = recs
        self.stores = stores
        self.a2i = {a: i for i, a in enumerate(vocab)}
        self.n_ans = len(vocab)
        # resolve (store, row) for every record once
        self.loc = []
        for r in recs:
            for si, s in enumerate(stores):
                if r["image"] in s.index:
                    self.loc.append((si, s.index[r["image"]]))
                    break
            else:
                raise KeyError(f"image {r['image']} not found in any feature store")
        enc = tokenizer([r["question"] for r in recs], padding="max_length", truncation=True,
                        max_length=max_q_len, return_tensors="np")
        self.input_ids = enc["input_ids"].astype(np.int64)
        self.attn = enc["attention_mask"].astype(np.int64)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        si, row = self.loc[i]
        st = self.stores[si]
        target = torch.zeros(self.n_ans)
        if r["answers"]:
            for a, s in soft_scores(r["answers"]).items():
                j = self.a2i.get(a)
                if j is not None:
                    target[j] = s
        ans_label = -1.0 if r["answerable"] is None else float(r["answerable"])
        type_label = TYPE2ID.get(r["answer_type"], -100) if r["answer_type"] else -100
        return {
            "idx": i,
            "patch": torch.from_numpy(np.asarray(st.patch[row], dtype=np.float32)),
            "img_global": torch.from_numpy(np.asarray(st.glob[row], dtype=np.float32)),
            "input_ids": torch.from_numpy(self.input_ids[i]),
            "attention_mask": torch.from_numpy(self.attn[i]),
            "target": target,
            "answerable": torch.tensor(ans_label),
            "atype": torch.tensor(type_label, dtype=torch.long),
        }
