"""Build one clean RAGTruth file in LettuceDetect's format, plus summary stats.

Input:  the two original files from https://github.com/ParticleMedia/RAGTruth/tree/main/dataset
        (response.jsonl, source_info.jsonl)
Output: ragtruth_data.json   -- one entry per answer (see README.md for fields)
        ragtruth_stats.json  -- counts, hallucination rates, lengths

Choices (see README.md):
  * prompt = RAGTruth's original `prompt` field, unchanged (same as LettuceDetect)
  * every answer and every label is kept (same as LettuceDetect)
  * dev set = 10% of training *prompts* per task, all 6 answers of a prompt kept together,
    random seed 42 (stricter than LettuceDetect's random 10% of answers)

Usage:
  python prepare_ragtruth.py --input_dir RAGTruth/dataset --output_dir .
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DEV_RATIO = 0.10
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_dev_sources(responses: list[dict], sources: dict[str, dict]) -> set[str]:
    """Pick 10% of training prompts per task. Returns the set of dev source_ids."""
    train_sources_by_task = defaultdict(set)
    for r in responses:
        if r["split"] == "train":
            train_sources_by_task[sources[r["source_id"]]["task_type"]].add(r["source_id"])
    rng = random.Random(SEED)
    dev = set()
    for task in sorted(train_sources_by_task):
        ids = sorted(train_sources_by_task[task])  # sort first so the result is reproducible
        rng.shuffle(ids)
        dev.update(ids[: round(len(ids) * DEV_RATIO)])
    return dev


def build_samples(responses: list[dict], sources: dict[str, dict], dev_sources: set[str]) -> list[dict]:
    samples = []
    for r in responses:
        s = sources[r["source_id"]]
        split = r["split"]
        if split == "train" and r["source_id"] in dev_sources:
            split = "dev"
        labels = [
            {
                "start": lab["start"],
                "end": lab["end"],
                "label": lab["label_type"],
                "implicit_true": lab["implicit_true"],
                "due_to_null": lab["due_to_null"],
            }
            for lab in r["labels"]
        ]
        samples.append(
            {
                "id": r["id"],
                "source_id": r["source_id"],
                "prompt": s["prompt"],
                "answer": r["response"],
                "labels": labels,
                "split": split,
                "task_type": s["task_type"],
                "model": r["model"],
                "quality": r["quality"],
            }
        )
    return samples


def check_samples(samples: list[dict]) -> None:
    """Sanity checks; raises if anything is wrong."""
    # every label position must lie inside the answer
    for x in samples:
        for lab in x["labels"]:
            assert 0 <= lab["start"] < lab["end"] <= len(x["answer"]), x["id"]
    # no prompt may appear in two splits
    splits_per_source = defaultdict(set)
    for x in samples:
        splits_per_source[x["source_id"]].add(x["split"])
    leaks = [sid for sid, sp in splits_per_source.items() if len(sp) > 1]
    assert not leaks, f"{len(leaks)} prompts appear in more than one split"


def compute_stats(samples: list[dict]) -> dict:
    stats = {"counts": {}, "response_level_rate": {}, "char_level_rate": {}, "answer_chars_mean": {},
             "prompt_chars_mean": {}, "label_types": {}, "quality": {}}
    groups = defaultdict(list)
    for x in samples:
        groups[(x["split"], x["task_type"])].append(x)
        groups[(x["split"], "ALL")].append(x)
    for (split, task), xs in sorted(groups.items()):
        key = f"{split}/{task}"
        n = len(xs)
        n_hal = sum(1 for x in xs if x["labels"])
        hal_chars = 0
        total_chars = 0
        for x in xs:
            mask = [False] * len(x["answer"])
            for lab in x["labels"]:
                for i in range(lab["start"], lab["end"]):
                    mask[i] = True
            hal_chars += sum(mask)
            total_chars += len(mask)
        stats["counts"][key] = n
        stats["response_level_rate"][key] = round(n_hal / n, 4)
        stats["char_level_rate"][key] = round(hal_chars / total_chars, 4)
        stats["answer_chars_mean"][key] = round(sum(len(x["answer"]) for x in xs) / n, 1)
        stats["prompt_chars_mean"][key] = round(sum(len(x["prompt"]) for x in xs) / n, 1)
    for split in ("train", "dev", "test"):
        xs = [x for x in samples if x["split"] == split]
        stats["label_types"][split] = dict(Counter(lab["label"] for x in xs for lab in x["labels"]))
        stats["quality"][split] = dict(Counter(x["quality"] for x in xs))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", default=Path("."), type=Path)
    args = ap.parse_args()

    responses = load_jsonl(args.input_dir / "response.jsonl")
    sources = {s["source_id"]: s for s in load_jsonl(args.input_dir / "source_info.jsonl")}

    dev_sources = choose_dev_sources(responses, sources)
    samples = build_samples(responses, sources, dev_sources)
    check_samples(samples)
    stats = compute_stats(samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ragtruth_data.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "ragtruth_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples")
    print(json.dumps({k: stats[k] for k in ("counts", "response_level_rate")}, indent=2))


if __name__ == "__main__":
    main()
