"""Efficiency benchmark: parameters, GFLOPs, model size, and batch-1 latency of the full pipeline
(raw image + question -> answer) on GPU (FP16), CPU (FP32) and CPU (INT8 dynamic quantisation).

  python src/benchmark.py --ours viz_lace=runs/viz_lace_s0 viz_clipmlp=runs/viz_clipmlp_s0 \
      --baselines vilt blip --out results/benchmark.json
"""
import argparse
import io
import json
import os
import platform
import statistics
import time

import numpy as np
import torch
from PIL import Image

from eval_e2e import BlipRunner, HF_DEFAULTS, OursRunner, ViltRunner

QUESTION = "What is the expiration date on this can of soup?"


def sample_image(seed=0, size=(640, 480)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def time_runner(runner, device, warmup, iters):
    img = sample_image()
    for _ in range(warmup):
        runner([img], [QUESTION])
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        runner([img], [QUESTION])
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": statistics.median(ts), "p90_ms": float(np.percentile(ts, 90)),
            "mean_ms": statistics.mean(ts)}


def count_flops(runner):
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        runner([sample_image()], [QUESTION])
    return fc.get_total_flops() / 1e9


def make_runner(kind, a, device, int8, novision=False):
    if kind in a.ours_map:
        return OursRunner(a.ours_map[kind], device, int8, novision)
    name = (a.vilt_model if kind == "vilt" else a.blip_model) or HF_DEFAULTS[kind]
    return (ViltRunner if kind == "vilt" else BlipRunner)(name, device, int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="*", default=[], metavar="NAME=RUN_DIR",
                    help="trained models to benchmark, e.g. viz_lace=runs/viz_lace_s0")
    ap.add_argument("--baselines", nargs="*", default=["vilt", "blip"])
    ap.add_argument("--vilt_model", default=None); ap.add_argument("--blip_model", default=None)
    ap.add_argument("--cpu_threads", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no_gpu", action="store_true"); ap.add_argument("--no_int8", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.ours_map = dict(x.split("=", 1) for x in a.ours)
    models = list(a.ours_map) + a.baselines

    env = {"torch": torch.__version__, "cpu": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    results = {"env": env, "models": {}}
    for kind in models:
        r = {}
        cpu = torch.device("cpu")
        base = make_runner(kind, a, cpu, False)
        model = base.model
        r["params_total_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        if kind in a.ours_map:
            r["params_head_M"] = sum(p.numel() for p in model.head.parameters()) / 1e6
        r["size_fp32_MB"] = size_mb(model)
        torch.set_num_threads(max(a.cpu_threads))
        r["gflops"] = count_flops(base)
        for th in a.cpu_threads:
            torch.set_num_threads(th)
            r[f"cpu_fp32_{th}t"] = time_runner(base, cpu, a.warmup, a.iters)
        del base, model
        if not a.no_int8:
            q = make_runner(kind, a, cpu, True)
            r["size_int8_MB"] = size_mb(q.model)
            for th in a.cpu_threads:
                torch.set_num_threads(th)
                r[f"cpu_int8_{th}t"] = time_runner(q, cpu, a.warmup, a.iters)
            del q
            if kind in a.ours_map:  # INT8 with the vision encoder kept in FP32
                q = make_runner(kind, a, cpu, True, novision=True)
                r["size_int8nv_MB"] = size_mb(q.model)
                for th in a.cpu_threads:
                    torch.set_num_threads(th)
                    r[f"cpu_int8nv_{th}t"] = time_runner(q, cpu, a.warmup, a.iters)
                del q
        if torch.cuda.is_available() and not a.no_gpu:
            gpu = torch.device("cuda")
            g = make_runner(kind, a, gpu, False)
            orig = g.__call__  # FP16 compute via autocast (weights stay FP32)

            def call_fp16(imgs, qs, _orig=orig):
                with torch.autocast("cuda", dtype=torch.float16):
                    return _orig(imgs, qs)
            torch.cuda.reset_peak_memory_stats()
            r["gpu_fp16"] = time_runner(call_fp16, gpu, a.warmup, a.iters)
            r["gpu_peak_mem_MB"] = torch.cuda.max_memory_allocated() / 1e6
            del g
            torch.cuda.empty_cache()
        results["models"][kind] = r
        print(kind, json.dumps(r, indent=1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
