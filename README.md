# NLP Hallucination Detection

COMS 4705 Natural Language Processing — Fall 2026

## Overview

This project investigates whether source-grounded hallucination detectors rely on task-specific shortcuts rather than retrieved evidence, and whether a Product-of-Experts (PoE) approach can improve generalization.

We use **RAGTruth** to train and evaluate token-level hallucination detectors based on ModernBERT, following the general methodology of LettuceDetect.

Our experiments compare a standard detector against several baselines, rebalanced training, and a proposed PoE detector.

## Project Structure

```text
src/
├── models/          # Model definitions (standard, PoE, baselines)
├── train.py         # Model training utilities
└── metrics.py       # Evaluation metrics

scripts/             # Training and evaluation entrypoints
tests/               # Unit and integration tests
results/             # Experiment outputs and metrics
checkpoints/         # Saved model checkpoints
```

*The structure above reflects the intended organization; some components are still in development.*

## Setup

Create and activate a Python virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Data Format

We use RAGTruth, which provides generated answers and annotated hallucination spans.

### Raw Examples

RAGTruth provides generated responses with annotated hallucination spans.

Preprocessing should preserve the information needed for:
- Standard detector training (prompt, retrieved context, answer, hallucination spans)
- Task-only baselines and rebalancing (task type)
- Evaluation and dataset splitting (example identifiers and metadata)

### Tokenized Examples

The standard detector's training dataset should provide:

| Field | Description |
|---|---|
| `input_ids` | Tokenized model input |
| `attention_mask` | Attention mask |
| `labels` | Token-level hallucination labels |

**Label conventions:**

- `0` — Supported / not hallucinated
- `1` — Hallucinated
- `-100` — Ignored during training and evaluation

Only answer tokens receive classification labels. Context and special tokens receive `-100`.

Hallucination character spans are converted to token labels using tokenizer offset mappings.

The standard detector does not receive `task_type` as an explicit input. However,
