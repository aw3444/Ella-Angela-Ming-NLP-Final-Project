
from datasets import Dataset
from transformers import AutoTokenizer

from src.train import train_standard_detector


MODEL_NAME = "answerdotai/ModernBERT-base"


def make_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    examples = [
        (
            "Paris is the capital of France.",
            "Paris is the capital.",
            False,
        ),
        (
            "Tokyo is the capital of Japan.",
            "London is the capital.",
            True,
        ),
        (
            "Water freezes at 0 degrees Celsius.",
            "Water freezes at 0 degrees Celsius.",
            False,
        ),
        (
            "Earth orbits the Sun.",
            "Earth orbits Jupiter.",
            True,
        ),
    ]

    processed = []

    for context, answer, hallucinated in examples:
        encoding = tokenizer(
            context,
            answer,
            truncation="only_first",
            max_length=128,
        )

        labels = [
            (1 if hallucinated else 0) if sid == 1 else -100
            for sid in encoding.sequence_ids()
        ]

        processed.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        })

    return Dataset.from_list(processed)


def test_standard_training():
    dataset = make_dataset()

    trainer = train_standard_detector(
        train_dataset=dataset,
        val_dataset=dataset,
        max_steps=1,
    )

    assert trainer.state.global_step == 1

    metrics = trainer.evaluate()

    assert "eval_loss" in metrics
    assert "eval_f1" in metrics
    assert "eval_precision" in metrics
    assert "eval_recall" in metrics
