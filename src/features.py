"""Extract and cache frozen CLIP image features (patch grid + global embedding) to .npy memmaps.

Example:
  python src/features.py --image_dir data/vizwiz/train --out feats/vizwiz_train \
      --backbone openai/clip-vit-base-patch16 --pool_grid 7
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from backbone import ClipBackbone, load_rgb
import corruptions


class ImageFolderList(Dataset):
    def __init__(self, image_dir, files, transform, corruption=None, severity=0):
        self.image_dir, self.files, self.transform = image_dir, files, transform
        self.corruption, self.severity = corruption, severity

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        try:
            img = load_rgb(os.path.join(self.image_dir, f))
        except Exception as e:  # corrupted file -> black image, logged
            print(f"[warn] could not read {f}: {e}")
            img = None
        if img is None:
            from PIL import Image
            img = Image.new("RGB", (224, 224))
        img = corruptions.apply(img, self.corruption, self.severity, seed=i)
        return self.transform(img), i


def list_images(image_dir, files_txt=None):
    if files_txt:
        with open(files_txt) as f:
            return [l.strip() for l in f if l.strip()]
    exts = (".jpg", ".jpeg", ".png")
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith(exts))


def extract(backbone, image_dir, out_prefix, batch_size=128, workers=4, device="cuda",
            files=None, corruption=None, severity=0, fp16=True):
    files = files if files is not None else list_images(image_dir)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    N = len(files)
    patch = np.lib.format.open_memmap(out_prefix + "_patch.npy", mode="w+", dtype=np.float16,
                                      shape=(N, backbone.n_img_tokens, backbone.dim_vision))
    glob = np.lib.format.open_memmap(out_prefix + "_global.npy", mode="w+", dtype=np.float16,
                                     shape=(N, backbone.dim_proj))
    ds = ImageFolderList(image_dir, files, backbone.transform, corruption, severity)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=(device == "cuda"))
    backbone.to(device)
    use_amp = fp16 and device == "cuda"
    for x, idx in tqdm(dl, desc=f"features -> {os.path.basename(out_prefix)}"):
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            tok, g = backbone.encode_image(x)
        idx = idx.numpy()
        patch[idx] = tok.float().cpu().numpy().astype(np.float16)
        glob[idx] = g.float().cpu().numpy().astype(np.float16)
    patch.flush()
    glob.flush()
    with open(out_prefix + "_index.json", "w") as f:
        json.dump({fn: i for i, fn in enumerate(files)}, f)
    with open(out_prefix + "_meta.json", "w") as f:
        json.dump({"backbone": getattr(backbone, "name", ""), "n": N, "tokens": backbone.n_img_tokens,
                   "dim_vision": backbone.dim_vision, "dim_proj": backbone.dim_proj,
                   "corruption": corruption, "severity": severity}, f)
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. feats/vizwiz_train")
    ap.add_argument("--backbone", default="openai/clip-vit-base-patch16")
    ap.add_argument("--pool_grid", type=int, default=7)
    ap.add_argument("--files_txt", default=None, help="optional list of image filenames to extract")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corruption", default=None)
    ap.add_argument("--severity", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    bb = ClipBackbone(a.backbone, pool_grid=a.pool_grid)
    bb.name = a.backbone
    files = list_images(a.image_dir, a.files_txt)
    n = extract(bb, a.image_dir, a.out, a.batch_size, a.workers, a.device, files, a.corruption, a.severity)
    print(f"wrote {n} images to {a.out}_*.npy")


if __name__ == "__main__":
    main()
