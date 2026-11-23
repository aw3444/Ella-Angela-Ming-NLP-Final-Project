"""Evaluate any Hugging Face-compatible binary token hallucination detector.

Run from repository root with: python -m scripts.evaluate_model --help
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data.data_utils import encode_example
from data.evaluate import (
    bootstrap_ci, evaluate, format_table, prediction_from_encoding, tune_threshold,
)


def predict_samples(model, tokenizer, samples, *, answer_only=False, batch_size=2):
    """Predict answer-token probabilities; pad batches to reduce GPU overhead."""
    predictions = {}
    model.eval()
    device = next(model.parameters()).device

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        encodings = [encode_example(tokenizer, x["prompt"], x["answer"], x["labels"],
                                    answer_only=answer_only) for x in batch]
        inputs = tokenizer.pad(
            [{"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
             for enc in encodings],
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)[..., 1].cpu()

        for sample, enc, row in zip(batch, encodings, probs):
            length = len(enc["input_ids"])
            predictions[sample["id"]] = prediction_from_encoding(enc, row[:length].tolist())

        if (start + len(batch)) % 100 < batch_size or start + len(batch) == len(samples):
            print(f"Predicted {start + len(batch)}/{len(samples)}", flush=True)

    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="checkpoints/standard/final",
                        help="Local model directory or Hugging Face model ID")
    parser.add_argument("--data", default="data/processed/ragtruth_data.json")
    parser.add_argument("--split", choices=["dev", "test", "train"], default="test")
    parser.add_argument("--answer_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune_on_dev", action="store_true",
                        help="Tune response-level F1 threshold on development predictions")
    parser.add_argument("--tune_metric", choices=["response", "span"], default="response")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--output_dir", default="results/standard")
    args = parser.parse_args()

    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch_size and --limit must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.tune_on_dev and args.split == "train":
        parser.error("Do not tune a threshold on dev while reporting train metrics")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(device)
    if model.config.num_labels != 2:
        raise ValueError(f"Expected 2 output labels; found {model.config.num_labels}")

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    samples = [x for x in data if x["split"] == args.split]
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        parser.error("No examples found for requested split")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.split if args.limit is None else f"{args.split}_first{args.limit}"
    if args.answer_only:
        run_name += "_answer_only"

    predictions = predict_samples(model, tokenizer, samples,
                                  answer_only=args.answer_only, batch_size=args.batch_size)
    pred_path = output_dir / f"{run_name}_preds.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    threshold = args.threshold
    if args.tune_on_dev:
        if args.split == "dev" and args.limit is None:
            dev_samples, dev_preds = samples, predictions
        else:
            # Tune on the full development set, even if test --limit is specified.
            dev_samples = [x for x in data if x["split"] == "dev"]
            dev_preds = predict_samples(model, tokenizer, dev_samples,
                                        answer_only=args.answer_only, batch_size=args.batch_size)
            dev_suffix = "_answer_only" if args.answer_only else ""
            (output_dir / f"dev{dev_suffix}_preds.json").write_text(
                json.dumps(dev_preds), encoding="utf-8")
        threshold = tune_threshold(dev_samples, dev_preds, metric=args.tune_metric)
        print(f"Dev-tuned {args.tune_metric} threshold: {threshold:.2f}")

    results = evaluate(samples, predictions, threshold=threshold)
    results["model_path"] = args.model_path
    results["answer_only"] = args.answer_only
    results["split"] = args.split
    if args.bootstrap:
        results["bootstrap_95ci"] = bootstrap_ci(samples, predictions, threshold)
    print(format_table(results))
    results_path = output_dir / f"{run_name}_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {pred_path} and {results_path}")


if __name__ == "__main__":
    main()
