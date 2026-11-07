"""Build a tiny synthetic VizWiz / VQA v2 copy and tiny random-weight CLIP / ViLT / BLIP models,
so the full pipeline can be smoke-tested offline on CPU. Numbers produced from this are meaningless.

  python tests/make_synthetic.py --out /tmp/smoke
"""
import argparse
import json
import os
import random
import string

import numpy as np
from PIL import Image

COLORS = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220), "yellow": (230, 220, 30)}


def make_images(dirpath, names, rng):
    os.makedirs(dirpath, exist_ok=True)
    meta = {}
    for n in names:
        cname = rng.choice(list(COLORS))
        w, h = rng.randint(40, 90), rng.randint(40, 90)
        arr = np.full((h, w, 3), COLORS[cname], np.uint8)
        k = rng.randint(1, 4)
        for _ in range(k):  # k white squares -> "how many"
            x, y = rng.randint(0, w - 8), rng.randint(0, h - 8)
            arr[y:y + 6, x:x + 6] = 255
        dark = rng.random() < 0.25
        if dark:  # "unanswerable" style image
            arr = (arr * 0.05).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(dirpath, n), quality=90)
        meta[n] = (cname, k, dark)
    return meta


def qa_for(meta, rng):
    cname, k, dark = meta
    kind = rng.choice(["color", "count", "yn"])
    if dark:
        return "what is this?", ["unanswerable"] * 7 + ["unsuitable"] * 3, "unanswerable", 0
    if kind == "color":
        return "what color is the background?", [cname] * 8 + [rng.choice(list(COLORS))] * 2, "other", 1
    if kind == "count":
        return "how many white squares are there?", [str(k)] * 9 + [str(k + 1)], "number", 1
    yes = rng.random() < 0.5
    c = cname if yes else rng.choice([c for c in COLORS if c != cname])
    return f"is the background {c}?", (["yes"] if yes else ["no"]) * 10, "yes/no", 1


def vizwiz_split(root, split, n, rng):
    names = [f"VizWiz_{split}_{i:08d}.jpg" for i in range(n)]
    meta = make_images(os.path.join(root, split), names, rng)
    items = []
    for nm in names:
        q, ans, t, a = qa_for(meta[nm], rng)
        items.append({"image": nm, "question": q, "answers": [{"answer": x, "answer_confidence": "yes"} for x in ans],
                      "answer_type": t, "answerable": a})
    os.makedirs(os.path.join(root, "Annotations"), exist_ok=True)
    json.dump(items, open(os.path.join(root, "Annotations", f"{split}.json"), "w"))


def vqav2_split(root, coco_root, split, n, rng, start_id):
    names = [f"COCO_{split}2014_{start_id + i:012d}.jpg" for i in range(n)]
    meta = make_images(os.path.join(coco_root, f"{split}2014"), names, rng)
    qs, anns = [], []
    for i, nm in enumerate(names):
        q, ans, t, _ = qa_for((meta[nm][0], meta[nm][1], False), rng)
        qid = (start_id + i) * 10
        qs.append({"image_id": start_id + i, "question": q, "question_id": qid})
        anns.append({"question_id": qid, "image_id": start_id + i, "answer_type": t, "question_type": "what",
                     "multiple_choice_answer": ans[0],
                     "answers": [{"answer": x, "answer_confidence": "yes", "answer_id": j + 1} for j, x in enumerate(ans)]})
    os.makedirs(root, exist_ok=True)
    json.dump({"questions": qs}, open(os.path.join(root, f"v2_OpenEnded_mscoco_{split}2014_questions.json"), "w"))
    json.dump({"annotations": anns}, open(os.path.join(root, f"v2_mscoco_{split}2014_annotations.json"), "w"))


WORDS = ["what", "color", "is", "the", "background", "how", "many", "white", "squares", "are", "there", "this",
         "red", "green", "blue", "yellow", "yes", "no", "unanswerable", "unsuitable", "of", "on", "can", "soup",
         "expiration", "date", "a"] + [str(i) for i in range(10)]


def tiny_clip(path):
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizer
    chars = list(string.ascii_lowercase + string.digits + "?.,'!-")
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for c in chars:
        vocab.setdefault(c, len(vocab))
        vocab.setdefault(c + "</w>", len(vocab))
    os.makedirs(path, exist_ok=True)
    json.dump(vocab, open(os.path.join(path, "vocab.json"), "w"))
    open(os.path.join(path, "merges.txt"), "w").write("#version: 0.2\n")
    tok = CLIPTokenizer(os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt"))
    cfg = CLIPConfig(
        text_config=dict(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, max_position_embeddings=64, bos_token_id=0, eos_token_id=1,
                         pad_token_id=1),
        vision_config=dict(image_size=32, patch_size=8, hidden_size=48, intermediate_size=96,
                           num_hidden_layers=2, num_attention_heads=2),
        projection_dim=24)
    CLIPModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)


def bert_tokenizer(path, extra=()):
    from transformers import BertTokenizerFast
    os.makedirs(path, exist_ok=True)
    toks = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *extra, *WORDS, "?"]
    open(os.path.join(path, "vocab.txt"), "w").write("\n".join(toks) + "\n")
    return BertTokenizerFast(os.path.join(path, "vocab.txt"), do_lower_case=True), toks


def tiny_vilt(path):
    from transformers import ViltConfig, ViltForQuestionAnswering, ViltImageProcessor, ViltProcessor
    tok, toks = bert_tokenizer(path)
    labels = ["red", "green", "blue", "yellow", "yes", "no", "1", "2", "3", "4"]
    cfg = ViltConfig(vocab_size=len(toks), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=40, image_size=64, patch_size=32,
                     max_image_length=-1, num_labels=len(labels),
                     id2label={i: l for i, l in enumerate(labels)}, label2id={l: i for i, l in enumerate(labels)})
    ViltForQuestionAnswering(cfg).save_pretrained(path)
    ViltProcessor(ViltImageProcessor(size={"shortest_edge": 64}, size_divisor=32), tok).save_pretrained(path)


def tiny_blip(path):
    from transformers import BlipConfig, BlipForQuestionAnswering, BlipImageProcessor, BlipProcessor
    tok, toks = bert_tokenizer(path, extra=["[DEC]"])
    dec = toks.index("[DEC]")
    cfg = BlipConfig(
        text_config=dict(vocab_size=len(toks), hidden_size=32, encoder_hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=64,
                         bos_token_id=dec, pad_token_id=0, sep_token_id=toks.index("[SEP]")),
        vision_config=dict(image_size=32, patch_size=8, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2),
        projection_dim=16)
    BlipForQuestionAnswering(cfg).save_pretrained(path)
    BlipProcessor(BlipImageProcessor(size={"height": 32, "width": 32}), tok).save_pretrained(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    d = os.path.join(a.out, "data")
    vizwiz_split(os.path.join(d, "vizwiz"), "train", 240, rng)
    vizwiz_split(os.path.join(d, "vizwiz"), "val", 80, rng)
    vqav2_split(os.path.join(d, "vqav2"), os.path.join(d, "coco"), "train", 160, rng, 1)
    vqav2_split(os.path.join(d, "vqav2"), os.path.join(d, "coco"), "val", 60, rng, 100000)
    tiny_clip(os.path.join(a.out, "models", "tiny-clip"))
    tiny_vilt(os.path.join(a.out, "models", "tiny-vilt"))
    tiny_blip(os.path.join(a.out, "models", "tiny-blip"))
    print("synthetic data + tiny models written to", a.out)


if __name__ == "__main__":
    main()
