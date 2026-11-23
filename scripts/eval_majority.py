
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from data.data_utils import encode_example
from data.evaluate import (
    evaluate,
    format_table,
    prediction_from_encoding,
)
from src.models.majority import MajorityBaseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/ragtruth_data.json",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
    )
    args = parser.parse_args()

    # Load original RAGTruth examples
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    train = [x for x in data if x["split"] == "train"]
    samples = [x for x in data if x["split"] == args.split]

    # Use the same tokenizer and preprocessing as ModernBERT
    tokenizer = AutoTokenizer.from_pretrained(
        "answerdotai/ModernBERT-base"
    )

    def encode(samples):
        return [
            encode_example(
                tokenizer,
                x["prompt"],
                x["answer"],
                x["labels"],
            )
            for x in samples
        ]

    train_encoded = encode(train)
    test_encoded = encode(samples)

    # Fit on training tokens only
    model = MajorityBaseline().fit(train_encoded)
    predicted_labels = model.predict(test_encoded)

    print(f"Majority token class: {model.majority_class}")

    # Convert token labels to the shared evaluation format
    preds = {}

    for example, enc, labels in zip(
        samples, test_encoded, predicted_labels
    ):
        probs = [
            float(label == 1)
            for label in labels
        ]

        preds[example["id"]] = prediction_from_encoding(
            enc, probs
        )

    # Evaluate against original character-level annotations
    results = evaluate(samples, preds)
    print(format_table(results))
    
    import numpy as np

    # Token-level accuracy (excluding ignored tokens)
    gold = np.concatenate([
        np.asarray(enc["labels"]) for enc in test_encoded
    ])
    pred = np.concatenate([
        np.asarray(labels) for labels in predicted_labels
    ])

    mask = gold != -100
    token_accuracy = np.mean(gold[mask] == pred[mask])

    # Response-level accuracy
    response_accuracy = np.mean([
        int(bool(x["labels"])) == model.majority_class
        for x in samples
    ])
    
    results["ALL"]["token_accuracy"] = float(token_accuracy)
    results["ALL"]["accuracy"] = float(response_accuracy)
    
    print(f"\nToken-level accuracy: {token_accuracy:.2%}")
    print(f"Response-level accuracy: {response_accuracy:.2%}")

    # Save metrics
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"majority_{args.split}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
