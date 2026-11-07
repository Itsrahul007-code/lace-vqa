"""Evaluation metrics: VQA accuracy (overall / per answer type), answerability AP, unanswerable F1."""
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from vqa_utils import normalize_answer, vqa_accuracy


def evaluate_predictions(answers: List[str], recs: List[Dict], p_answerable: Optional[np.ndarray] = None) -> Dict:
    accs, by_type = [], defaultdict(list)
    for a, r in zip(answers, recs):
        if not r["answers"]:
            continue
        acc = vqa_accuracy(normalize_answer(a), r["answers"], normalized=True)
        accs.append(acc)
        by_type[r["answer_type"] or "all"].append(acc)
    out = {"accuracy": float(np.mean(accs)) if accs else float("nan"), "n": len(accs),
           "by_type": {t: {"accuracy": float(np.mean(v)), "n": len(v)} for t, v in sorted(by_type.items())}}

    lab = np.array([r["answerable"] if r["answerable"] is not None else -1 for r in recs])
    m = lab >= 0
    if m.any() and len(set(lab[m])) == 2:
        from sklearn.metrics import average_precision_score, f1_score
        if p_answerable is not None:
            out["answerability_ap"] = float(average_precision_score(lab[m], np.asarray(p_answerable)[m]))
        pred_unans = np.array([normalize_answer(a) == "unanswerable" for a in answers])
        out["unanswerable_f1"] = float(f1_score(1 - lab[m], pred_unans[m].astype(int)))
        out["accuracy_answerable_only"] = float(np.mean([vqa_accuracy(normalize_answer(a), r["answers"], True)
                                                         for a, r in zip(answers, recs)
                                                         if r["answerable"] == 1 and r["answers"]]))
    return out
