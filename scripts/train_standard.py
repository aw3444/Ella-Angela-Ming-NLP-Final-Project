import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from data.data_utils import encode_example
from src.train import MODEL_NAME, train_standard_detector


def load_dataset(data_path, split, tokenizer, limit=None):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = [x for x in data if x["split"] == split]

    if limit is not None:
        examples = examples[:limit]

    encoded = []

    for example in examples:
        encoding = encode_example(
            tokenizer,
            example["prompt"],
            example["answer"],
            example["labels"],
        )

        encoded.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
        })

    return Dataset.from_list(encoded)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/processed/ragtruth_data.json",
    )
    parser.add_argument(
        "--experiment",
        choices=["standard"],
        default="standard",
    )
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = load_dataset(
        args.data, "train", tokenizer, args.train_limit
    )
    val_dataset = load_dataset(
        args.data, "dev", tokenizer, args.val_limit
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples: {len(val_dataset)}")

    output_dir = Path("checkpoints") / args.experiment

    trainer = train_standard_detector(
        train_dataset,
        val_dataset,
        max_steps=args.max_steps,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
    )

    metrics = trainer.evaluate()
    print("Final validation metrics:", metrics)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
