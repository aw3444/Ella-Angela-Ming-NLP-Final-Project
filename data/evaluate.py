"""Shared scoring code. Every model's numbers should come from here.

What each model has to provide, per answer (keyed by the answer's `id` in ragtruth_data.json):

  Token-level models (standard / PoE / answer-only detectors, LettuceDetect):
      {"token_probs": [p1, p2, ...],          # P(hallucinated) for each ANSWER token
       "offsets":     [[s1, e1], [s2, e2], ...]}  # character span of each answer token in `answer`
      -> use `prediction_from_encoding(enc, probs)` to build this from encode_example's output.

  Response-level-only models (majority class, task-only):
      {"response_score": p}                    # P(answer contains a hallucination)
      -> span-level scores are skipped for these.

Metrics (definitions follow LettuceDetect's evaluation code, plus AUROC from our proposal):
  * Response level: an answer is flagged if any answer token has prob > threshold
    (or response_score > threshold). Precision / recall / F1 of the "hallucinated" class.
  * Span level: flagged tokens next to each other are merged into character spans.
    precision = overlapping characters / predicted characters,
    recall    = overlapping characters / true hallucinated characters,
    both summed over all answers in the group.
  * AUROC: each answer is scored by its highest token prob (or response_score). Does not
    depend on the threshold.
  Every metric is reported pooled ("ALL"), per task, and as the average of the three
  tasks ("TASK_MACRO").

Usage from the command line:
  python evaluate.py --data ragtruth_data.json --preds test_preds.json --split test
  python evaluate.py --data ragtruth_data.json --preds test_preds.json --split test \
                     --tune_preds dev_preds.json          # pick the threshold on dev first
"""

import argparse
import json
import random

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

TASKS = ("QA", "Summary", "Data2txt")


# ---------------------------------------------------------------- building predictions

def prediction_from_encoding(enc: dict, probs) -> dict:
    """Turn a model's per-position probabilities into the format above.

    enc   -- the dict returned by data_utils.encode_example for this answer
    probs -- P(hallucinated) for every position of enc["input_ids"] (same length)
    """
    probs = list(map(float, probs))
    assert len(probs) == len(enc["input_ids"]), "probs must have one value per input position"
    token_probs, offsets = [], []
    for p, off in zip(probs, enc["answer_offsets"]):
        if off is not None:
            token_probs.append(p)
            offsets.append([off[0], off[1]])
    return {"token_probs": token_probs, "offsets": offsets}


def predicted_spans(pred: dict, threshold: float) -> list[tuple[int, int]]:
    """Merge neighbouring flagged answer tokens into character spans."""
    spans = []
    current = None
    for p, (s, e) in zip(pred["token_probs"], pred["offsets"]):
        if e <= s:  # empty-width tokens never start or extend a span
            continue
        if p > threshold:
            if current is None:
                current = [s, e]
            else:
                current[1] = e
        elif current is not None:
            spans.append(tuple(current))
            current = None
    if current is not None:
        spans.append(tuple(current))
    return spans


def response_score(pred: dict) -> float:
    if "response_score" in pred:
        return float(pred["response_score"])
    return max(pred["token_probs"], default=0.0)


# ---------------------------------------------------------------- metrics

def _char_mask(spans, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for s, e in spans:
        mask[max(0, s):min(length, e)] = True
    return mask


def _prf(tp: float, n_pred: float, n_gold: float) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def score_group(samples: list[dict], preds: dict, threshold: float) -> dict:
    """All metrics for one group of answers (e.g. all test answers, or one task)."""
    y_true = [1 if x["labels"] else 0 for x in samples]
    scores = [response_score(preds[x["id"]]) for x in samples]
    y_pred = [1 if s > threshold else 0 for s in scores]

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None, zero_division=0)
    out = {
        "n": len(samples),
        "n_hallucinated": int(sum(y_true)),
        "response": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0])},
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else None,
    }

    if all("token_probs" in preds[x["id"]] for x in samples):
        overlap = n_pred = n_gold = 0
        for x in samples:
            length = len(x["answer"])
            gold = _char_mask([(lab["start"], lab["end"]) for lab in x["labels"]], length)
            pred = _char_mask(predicted_spans(preds[x["id"]], threshold), length)
            overlap += int((gold & pred).sum())
            n_pred += int(pred.sum())
            n_gold += int(gold.sum())
        out["span"] = _prf(overlap, n_pred, n_gold)
    else:
        out["span"] = None
    return out


def evaluate(samples: list[dict], preds: dict, threshold: float = 0.5) -> dict:
    """Pooled, per-task, and task-averaged metrics."""
    missing = [x["id"] for x in samples if x["id"] not in preds]
    assert not missing, f"{len(missing)} answers have no prediction, e.g. {missing[:3]}"

    results = {"threshold": threshold, "ALL": score_group(samples, preds, threshold)}
    for task in TASKS:
        group = [x for x in samples if x["task_type"] == task]
        if group:
            results[task] = score_group(group, preds, threshold)

    present = [t for t in TASKS if t in results]
    macro = {}
    for key in ("response", "span"):
        if all(results[t][key] is not None for t in present):
            macro[key] = {m: float(np.mean([results[t][key][m] for t in present]))
                          for m in ("precision", "recall", "f1")}
        else:
            macro[key] = None
    aucs = [results[t]["auroc"] for t in present]
    macro["auroc"] = float(np.mean(aucs)) if None not in aucs else None
    results["TASK_MACRO"] = macro
    return results


def tune_threshold(dev_samples: list[dict], dev_preds: dict, metric: str = "response") -> float:
    """Pick the threshold with the best pooled F1 on the dev set (metric: "response" or "span")."""
    best_t, best_f = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        group = score_group(dev_samples, dev_preds, float(t))
        f = group[metric]["f1"] if group[metric] else -1.0
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def bootstrap_ci(samples: list[dict], preds: dict, threshold: float = 0.5,
                 n_boot: int = 1000, seed: int = 0) -> dict:
    """95% confidence intervals for pooled response F1, span F1 and AUROC (resampling answers)."""
    rng = random.Random(seed)
    vals = {"response_f1": [], "span_f1": [], "auroc": []}
    for _ in range(n_boot):
        resample = [samples[rng.randrange(len(samples))] for _ in samples]
        g = score_group(resample, preds, threshold)
        vals["response_f1"].append(g["response"]["f1"])
        if g["span"]:
            vals["span_f1"].append(g["span"]["f1"])
        if g["auroc"] is not None:
            vals["auroc"].append(g["auroc"])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else None
            for k, v in vals.items()}


# ---------------------------------------------------------------- printing / CLI

def format_table(results: dict) -> str:
    def pct(v):
        return "  -  " if v is None else f"{100 * v:5.1f}"
    lines = [f"threshold = {results['threshold']}",
             f"{'group':11s} {'n':>5s} {'resp P':>7s} {'resp R':>7s} {'resp F1':>7s} "
             f"{'span P':>7s} {'span R':>7s} {'span F1':>7s} {'AUROC':>7s}"]
    for name in ("ALL",) + TASKS + ("TASK_MACRO",):
        if name not in results:
            continue
        g = results[name]
        resp, span = g["response"], g["span"]
        n = str(g.get("n", ""))
        lines.append(
            f"{name:11s} {n:>5s} {pct(resp and resp['precision']):>7s} {pct(resp and resp['recall']):>7s} "
            f"{pct(resp and resp['f1']):>7s} {pct(span and span['precision']):>7s} "
            f"{pct(span and span['recall']):>7s} {pct(span and span['f1']):>7s} {pct(g['auroc']):>7s}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True, help="JSON: {answer id: prediction}")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tune_preds", help="dev-set predictions; if given, the threshold is chosen on dev")
    ap.add_argument("--bootstrap", action="store_true", help="also print 95%% confidence intervals")
    ap.add_argument("--out", help="write all results to this JSON file")
    args = ap.parse_args()

    data = json.load(open(args.data, encoding="utf-8"))
    preds = json.load(open(args.preds, encoding="utf-8"))
    samples = [x for x in data if x["split"] == args.split]

    threshold = args.threshold
    if args.tune_preds:
        dev = [x for x in data if x["split"] == "dev"]
        threshold = tune_threshold(dev, json.load(open(args.tune_preds, encoding="utf-8")))

    results = evaluate(samples, preds, threshold)
    print(format_table(results))
    if args.bootstrap:
        results["bootstrap_95ci"] = bootstrap_ci(samples, preds, threshold)
        print("95% CI (pooled):", results["bootstrap_95ci"])
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
