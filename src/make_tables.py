"""Collect all results into LaTeX macros (paper/results_macros.tex) and figures (paper/figures/).

Every training run directory is named <run>_s<seed>; values are reported as mean +- std over seeds.
The paper uses \\res{<key>} which prints [TBD] until the key exists, so tables fill automatically.

  python src/make_tables.py --runs runs --e2e results --bench results/benchmark.json --paper paper
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

TYPE_KEYS = {"yes/no": "yn", "number": "num", "other": "other", "unanswerable": "unans"}


def fmt(vals, scale=100.0, nd=2):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    m = np.mean(vals) * scale
    if len(vals) == 1:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}{{\\scriptsize$\\pm${np.std(vals, ddof=1) * scale:.{nd}f}}}"


def collect_runs(runs_dir):
    groups = defaultdict(list)
    for p in glob.glob(os.path.join(runs_dir, "*", "results.json")):
        name = os.path.basename(os.path.dirname(p))
        base = re.sub(r"_s\d+$", "", name)
        groups[base].append(json.load(open(p)))
    macros = {}
    for base, rs in groups.items():
        ev = [r["test"]["gated"] for r in rs]
        macros[f"{base}-acc"] = fmt([e["accuracy"] for e in ev])
        macros[f"{base}-argmaxacc"] = fmt([r["test"]["argmax"]["accuracy"] for r in rs])
        for t, k in TYPE_KEYS.items():
            macros[f"{base}-{k}"] = fmt([e["by_type"].get(t, {}).get("accuracy") for e in ev])
        macros[f"{base}-ap"] = fmt([e.get("answerability_ap") for e in ev])
        macros[f"{base}-f1"] = fmt([e.get("unanswerable_f1") for e in ev])
        macros[f"{base}-ansonly"] = fmt([e.get("accuracy_answerable_only") for e in ev])
        macros[f"{base}-params"] = f"{rs[0]['trainable_params'] / 1e6:.2f}"
        macros[f"{base}-traintime"] = f"{np.mean([r['train_time_s'] for r in rs]) / 60:.1f}"
        macros[f"{base}-nseeds"] = str(len(rs))
        macros[f"{base}-oracle"] = fmt([r["oracle_acc"]["test"] for r in rs])
    return macros, groups


def collect_e2e(e2e_dir):
    """Zero-shot files results/zs_<model>.json (full split) and robustness files
    results/rob_<model>__<corruption>_<severity>.json (fixed subset; severity 0 = clean)."""
    macros, table = {}, defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(e2e_dir, "zs_*.json")):
        r = json.load(open(p))
        model = os.path.basename(p)[3:-5]
        macros[f"zs-{model}-acc"] = fmt([r["accuracy"]])
        macros[f"zs-{model}-ansonly"] = fmt([r.get("accuracy_answerable_only")])
        for t, k in TYPE_KEYS.items():
            macros[f"zs-{model}-{k}"] = fmt([r["by_type"].get(t, {}).get("accuracy")])
    for p in glob.glob(os.path.join(e2e_dir, "rob_*.json")):
        r = json.load(open(p))
        model = os.path.basename(p)[4:-5].split("__")[0]
        table[model][r["corruption"]][r["severity"]] = r
    for model, by_c in table.items():
        clean = by_c.get("none", {}).get(0)
        if clean:
            macros[f"rob-{model}-clean"] = fmt([clean["accuracy"]])
            if clean.get("mean_p_answerable") is not None:
                macros[f"robpans-{model}-clean"] = fmt([clean["mean_p_answerable"]])
        means = []
        for c, by_s in by_c.items():
            if c == "none":
                continue
            m = np.mean([v["accuracy"] for v in by_s.values()])
            means.append(m)
            macros[f"rob-{model}-{c.replace('_', '')}"] = fmt([m])
            if clean and clean["accuracy"] > 0:
                macros[f"robrel-{model}-{c.replace('_', '')}"] = fmt([m / clean["accuracy"]], nd=1)
            pa = [v.get("mean_p_answerable") for v in by_s.values() if v.get("mean_p_answerable") is not None]
            if pa:
                macros[f"robpans-{model}-{c.replace('_', '')}"] = fmt([np.mean(pa)])
        if means:
            macros[f"rob-{model}-mean"] = fmt([float(np.mean(means))])
            if clean and clean["accuracy"] > 0:
                macros[f"robrel-{model}-mean"] = fmt([float(np.mean(means)) / clean["accuracy"]], nd=1)
    for tag in ("int8nv", "int8", "fp32cpu"):
        for p in glob.glob(os.path.join(e2e_dir, f"{tag}_*.json")):
            r = json.load(open(p))
            run = os.path.basename(p)[len(tag) + 1:-5]
            macros[f"{tag}-{run}-acc"] = fmt([r["accuracy"]])
            macros[f"{tag}-{run}-qps"] = f"{r['throughput_qps']:.1f}"
    return macros, table


def per_question_acc(runs_dir, base):
    """Per-question VQA accuracy averaged over all seeds of a run group -> (qids, acc, types)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vqa_utils import normalize_answer, vqa_accuracy
    files = sorted(glob.glob(os.path.join(runs_dir, f"{base}_s*", "predictions_test.json")))
    if not files:
        return None
    accs, qids, types = [], None, None
    for f in files:
        preds = json.load(open(f))
        preds = [p for p in preds if p["gt"]]
        if qids is None:
            qids = [p["qid"] for p in preds]
            types = [p["answer_type"] for p in preds]
        d = {p["qid"]: vqa_accuracy(normalize_answer(p["pred"]), p["gt"], normalized=True) for p in preds}
        accs.append([d[q] for q in qids])
    return qids, np.mean(np.array(accs), 0), np.array(types)


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Two-sided paired bootstrap over questions: mean difference, 95% CI and p-value."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), min(p, 1.0)


def collect_significance(runs_dir, pairs=(("viz_lace", "viz_clipmlp"), ("viz_lace", "viz_lacev1"),
                                          ("viz_lace_tiny8m", "viz_clipmlp"))):
    macros, report = {}, []
    for a, b in pairs:
        ra, rb = per_question_acc(runs_dir, a), per_question_acc(runs_dir, b)
        if ra is None or rb is None or ra[0] != rb[0]:
            continue
        key = f"sig-{a}-vs-{b}"
        subsets = [("all", np.ones(len(ra[1]), bool))] + [(t, ra[2] == t) for t in sorted(set(ra[2]))]
        for name, m in subsets:
            diff, lo, hi, p = paired_bootstrap(ra[1][m], rb[1][m])
            k = key + ("" if name == "all" else "-" + {"yes/no": "yn", "number": "num", "other": "other",
                                                         "unanswerable": "unans"}.get(name, name))
            macros[k + "-diff"] = f"{diff*100:+.2f}"
            macros[k + "-ci"] = f"[{lo*100:+.2f}, {hi*100:+.2f}]"
            macros[k + "-p"] = "$<$0.001" if p < 0.001 else f"{p:.3f}"
            report.append(f"{a} vs {b} [{name}, n={int(m.sum())}]: diff {diff*100:+.2f} "
                          f"CI [{lo*100:+.2f}, {hi*100:+.2f}] p={p:.4f}")
    return macros, report


def collect_bench(path):
    macros = {}
    if not path or not os.path.exists(path):
        return macros
    b = json.load(open(path))
    for model, r in b["models"].items():
        macros[f"bench-{model}-params"] = f"{r['params_total_M']:.1f}"
        if "params_head_M" in r:
            macros[f"bench-{model}-headparams"] = f"{r['params_head_M']:.2f}"
        macros[f"bench-{model}-gflops"] = f"{r['gflops']:.1f}"
        macros[f"bench-{model}-size"] = f"{r['size_fp32_MB']:.0f}"
        if "size_int8_MB" in r:
            macros[f"bench-{model}-sizeint8"] = f"{r['size_int8_MB']:.0f}"
        if "size_int8nv_MB" in r:
            macros[f"bench-{model}-sizeint8nv"] = f"{r['size_int8nv_MB']:.0f}"
        for k, v in r.items():
            if isinstance(v, dict) and "median_ms" in v:
                macros[f"bench-{model}-{k.replace('_', '')}"] = f"{v['median_ms']:.0f}" if v["median_ms"] >= 10 \
                    else f"{v['median_ms']:.1f}"
        if "gpu_peak_mem_MB" in r:
            macros[f"bench-{model}-gpumem"] = f"{r['gpu_peak_mem_MB']:.0f}"
    macros["bench-gpu"] = (b["env"].get("gpu") or "n/a").replace("_", " ")
    macros["bench-cpu"] = f"{b['env']['cpu_count']}-core {b['env']['cpu']}".replace("_", " ")
    return macros


def plot_robustness(table, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"viz_lace": "LACE-B/16 (ours)", "viz_lace_tiny8m": "LACE-Tiny (ours)", "viz_clipmlp": "CLIP-MLP",
             "vilt": "ViLT (zero-shot)", "blip": "BLIP (zero-shot)"}
    corrs = sorted({c for m in table.values() for c in m if c != "none"})
    if not corrs:
        return False
    fig, axes = plt.subplots(2, (len(corrs) + 1) // 2, figsize=(3.0 * ((len(corrs) + 1) // 2), 5.2),
                             sharey=True, squeeze=False)
    for ax, c in zip(axes.flat, corrs):
        for model, by_c in sorted(table.items()):
            clean = by_c.get("none", {}).get(0)
            pts = sorted(by_c.get(c, {}).items())
            if not pts:
                continue
            xs = ([0] if clean else []) + [s for s, _ in pts]
            ys = ([clean["accuracy"] * 100] if clean else []) + [v["accuracy"] * 100 for _, v in pts]
            ax.plot(xs, ys, marker="o", lw=1.5, label=names.get(model, model))
        ax.set_title(c.replace("_", " ").capitalize(), fontsize=9)
        ax.set_xlabel("severity", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("VQA accuracy (%)", fontsize=8)
    for ax in list(axes.flat)[len(corrs):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_pdf)
    return True


def plot_curves(groups, out_pdf, runs_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    drawn = False
    for base in ["viz_clipmlp", "viz_lace", "viz_lace_vqapt"]:
        logs = glob.glob(os.path.join(runs_dir, f"{base}_s*", "log.jsonl"))
        if not logs:
            continue
        curves = [[json.loads(l)["dev_acc"] * 100 for l in open(p)] for p in logs]
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        x = np.arange(1, n + 1)
        ax.plot(x, arr.mean(0), label=base.replace("viz_", ""))
        if len(curves) > 1:
            ax.fill_between(x, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.2)
        drawn = True
    if drawn:
        ax.set_xlabel("epoch"); ax.set_ylabel("dev VQA accuracy (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_pdf)
    return drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--e2e", default="results")
    ap.add_argument("--bench", default="results/benchmark.json"); ap.add_argument("--paper", default="paper")
    a = ap.parse_args()
    m1, groups = collect_runs(a.runs)
    m2, table = collect_e2e(a.e2e)
    m3 = collect_bench(a.bench)
    m4, report = collect_significance(a.runs)
    print("\n".join(report))
    macros = {k: v for d in (m1, m2, m3, m4) for k, v in d.items() if v is not None}
    os.makedirs(os.path.join(a.paper, "figures"), exist_ok=True)
    with open(os.path.join(a.paper, "results_macros.tex"), "w") as f:
        f.write("% generated by src/make_tables.py -- do not edit by hand\n")
        for k in sorted(macros):
            f.write(f"\\expandafter\\def\\csname res@{k}\\endcsname{{{macros[k]}}}\n")
    with open(os.path.join(a.paper, "results_macros.json"), "w") as f:
        json.dump(macros, f, indent=1, sort_keys=True)
    r = plot_robustness(table, os.path.join(a.paper, "figures", "robustness.pdf"))
    c = plot_curves(groups, os.path.join(a.paper, "figures", "curves.pdf"), a.runs)
    print(f"{len(macros)} result macros written; robustness figure: {r}; curves figure: {c}")


if __name__ == "__main__":
    main()
