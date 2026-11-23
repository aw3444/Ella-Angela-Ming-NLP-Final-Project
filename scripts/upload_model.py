"""
Upload a trained model checkpoint to Hugging Face Hub.

Usage:
    python -m scripts.upload_model \
        --checkpoint outputs/standard/checkpoint-XXXX \
        --model standard \
        --username YOUR_USERNAME
"""

import argparse
from pathlib import Path

from huggingface_hub import create_repo
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)


MODEL_NAMES = {
    "standard": "ragtruth-modernbert-standard",
    "answer-only": "ragtruth-modernbert-answer-only",
    "poe": "ragtruth-modernbert-poe",
}


def upload_model(checkpoint, model_name, username, private=True):
    checkpoint = Path(checkpoint)

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint}"
        )

    repo_id = f"{username}/{MODEL_NAMES[model_name]}"

    print(f"Loading checkpoint: {checkpoint}")

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint
    )

    # All three models use the same ModernBERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        "answerdotai/ModernBERT-base"
    )

    print(f"Creating repository: {repo_id}")

    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    print("Uploading model...")
    model.push_to_hub(repo_id)

    print("Uploading tokenizer...")
    tokenizer.push_to_hub(repo_id)

    print(f"Successfully uploaded: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a model checkpoint to Hugging Face"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the best model checkpoint",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES.keys(),
        help="Model variant to upload",
    )

    parser.add_argument(
        "--username",
        required=True,
        help="Hugging Face username or organization",
    )

    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public (default: private)",
    )

    args = parser.parse_args()

    upload_model(
        checkpoint=args.checkpoint,
        model_name=args.model,
        username=args.username,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
