import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from data.evaluate import evaluate, format_table, tune_threshold
from src.models.task_only import TaskOnlyBaseline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/processed/ragtruth_data.json")
    parser.add_argument("--output_dir", default="results/task_only")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="e.g. answerdotai/ModernBERT-base; also computes token rates for PoE",
    )

    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    train_samples = [x for x in data if x["split"] == "train"]
    dev_samples = [x for x in data if x["split"] == "dev"]
    test_samples = [x for x in data if x["split"] == "test"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) if args.tokenizer else None
    model = TaskOnlyBaseline().fit(train_samples, tokenizer)

    dev_preds = model.predict(dev_samples)
    test_preds = model.predict(test_samples)

    # pick the threshold on dev, then score test
    threshold = tune_threshold(dev_samples, dev_preds)
    results = evaluate(test_samples, test_preds, threshold)
    print(format_table(results))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rates = {"response_rates": model.response_rates, "token_rates": model.token_rates}
    print("Training rates:", rates)

    (output_dir / "task_rates.json").write_text(json.dumps(rates, indent=2))
    (output_dir / "dev_preds.json").write_text(json.dumps(dev_preds))
    (output_dir / "test_preds.json").write_text(json.dumps(test_preds))
    (output_dir / "test_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
