from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.metrics import compute_metrics
from src.models.standard import build_standard_detector


MODEL_NAME = "answerdotai/ModernBERT-base"

def train_standard_detector(
    train_dataset,
    val_dataset,
    max_steps=-1,
    output_dir="checkpoints/standard",
    batch_size=4
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_standard_detector(MODEL_NAME)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        logging_steps=10,
        logging_first_step=True,
        learning_rate=1e-5,
        num_train_epochs=6,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=123,
        data_seed=123,
        report_to="none",
        max_steps=max_steps
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
        ),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer