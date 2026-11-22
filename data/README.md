# RAGTruth data for our project

Built from the **original** RAGTruth files (https://github.com/ParticleMedia/RAGTruth/tree/main/dataset),
not the wandb processed copy, so the prompts are exactly what LettuceDetect trained on.

## Files

| File | What it is |
|---|---|
| `ragtruth_data.json` | One entry per answer (17,790 total). Use this for everything. |
| `ragtruth_stats.json` | Counts, hallucination rates, average lengths, label types. |
| `prepare_ragtruth.py` | Rebuilds the two files above from the original data. |
| `data_utils.py` | `encode_example(...)`: the shared function that turns an entry into model inputs + token labels. **Everyone uses this.** |
| `check_lengths.py` | Counts input lengths with the real ModernBERT tokenizer. |
| `evaluate.py` | Shared scoring code: response-level and span-level precision/recall/F1 and AUROC, pooled, per task, and averaged over tasks. **Every model's numbers come from here.** |

## Fields in `ragtruth_data.json`

| Field | Meaning |
|---|---|
| `id` | RAGTruth's answer id |
| `source_id` | Which prompt the answer belongs to (each prompt has 6 answers, one per LLM) |
| `prompt` | RAGTruth's original prompt, unchanged (instruction + source) |
| `answer` | The LLM's answer |
| `labels` | Hallucinated spans: `start`, `end` (character positions in `answer`), `label` (e.g. "Evident Conflict"), plus RAGTruth's `implicit_true` and `due_to_null` flags |
| `split` | `train`, `dev`, or `test` |
| `task_type` | `QA`, `Summary`, or `Data2txt` |
| `model` | Which LLM wrote the answer |
| `quality` | `good`, `incorrect_refusal`, or `truncated` (RAGTruth's flag) |

An answer is "hallucinated" if `labels` is non-empty.

## Choices we made

1. **Prompt:** RAGTruth's original `prompt`, unchanged. Same as LettuceDetect.
2. **Nothing filtered:** every answer and every label is kept, including `implicit_true`,
   `due_to_null`, `truncated`, and `incorrect_refusal`. Same as LettuceDetect.
3. **Dev set:** 10% of training *prompts* in each task, chosen with random seed 42. All six answers to a
   prompt stay together, so no prompt is in both train and dev. *Differs from LettuceDetect*, which takes a
   random 10% of answers. The test set is RAGTruth's official test set, untouched.
4. **Inputs:** `[CLS] prompt [SEP] answer [SEP]`, ModernBERT tokenizer, max 4,096 tokens; if too long,
   only the prompt is cut. Prompt tokens get label -100; answer tokens get 1 if they overlap a
   hallucinated span, else 0. Same as LettuceDetect.

## Sizes and hallucination rates (share of answers with at least one hallucination)

| Split | QA | Summary | Data2txt | All |
|---|---|---|---|---|
| train (13,584) | 31.4% | 31.1% | 69.2% | 44.6% |
| dev (1,506) | 28.2% | 31.4% | 71.0% | 44.2% |
| test (2,700) | 17.8% | 22.7% | 64.3% | 34.9% |

Note: the **test set has fewer hallucinations than train**, especially for QA and summarization.
Thresholds tuned on dev may therefore flag too many test answers; AUROC is not affected.

## Example

```python
import json
from transformers import AutoTokenizer
from data_utils import encode_example

data = json.load(open("ragtruth_data.json"))
train = [x for x in data if x["split"] == "train"]
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

x = train[0]
enc = encode_example(tok, x["prompt"], x["answer"], x["labels"])                    # full detector
enc_ans = encode_example(tok, x["prompt"], x["answer"], x["labels"], answer_only=True)  # answer-only baseline
```

## Scoring models (`evaluate.py`)

Save each model's predictions as one JSON file `{answer id: prediction}`:

* **Token-level models** (standard, PoE, answer-only, LettuceDetect):
  `{"token_probs": [...], "offsets": [[start, end], ...]}`, one entry per **answer** token.
  Build it with `prediction_from_encoding(enc, probs)`, where `enc` is the output of
  `encode_example` and `probs` is P(hallucinated) for every input position.
* **Answer-level-only models** (majority class, task-only): `{"response_score": p}`.
  Span-level scores are left blank for these.

Then:

```
python evaluate.py --data ragtruth_data.json --preds test_preds.json --split test              # threshold 0.5
python evaluate.py --data ragtruth_data.json --preds test_preds.json --split test --tune_preds dev_preds.json
python evaluate.py ... --bootstrap --out results.json   # adds 95% confidence intervals, saves everything
```

What the numbers mean (same definitions as LettuceDetect's evaluation code):

* **Response level:** an answer is flagged if any answer token has probability > threshold.
  Precision / recall / F1 are for the "hallucinated" class.
* **Span level:** neighbouring flagged tokens are merged into character spans;
  precision = overlapping characters / predicted characters, recall = overlapping characters / true
  hallucinated characters, summed over all answers. (125 pairs of true spans overlap each other in
  RAGTruth; we count each character once, so numbers can differ very slightly from LettuceDetect's code.)
* **AUROC:** each answer is scored by its highest token probability; independent of the threshold.
* Rows: `ALL` (pooled), each task, and `TASK_MACRO` (average of the three tasks).

Checks run: perfect predictions give 100 on every metric; all-zero predictions give F1 0 and AUROC 0.5;
a hand-made overlap case gives the expected 0.5 / 0.5.

Two things the checks showed:

* **Task identity alone looks good when pooled.** The task-only baseline (predict each task's training
  hallucination rate) gets pooled AUROC 70.4 and response F1 62.8 on test, but task-averaged AUROC 50.0.
  This is why we report per-task and task-averaged scores.
* **Answer length can leak in.** Scoring an answer by its *highest* token probability favours longer answers
  (more tokens, more chances to be high). In QA, hallucinated test answers are longer on average than faithful
  ones (881 vs. 600 characters), so even random token probabilities give QA AUROC 0.585. Keep this in mind
  when reading QA AUROC.

## Not done yet

* Real token lengths: run `python check_lengths.py ragtruth_data.json` (needs Hugging Face access).
  The longest input is ~12,500 characters, so most likely everything fits in 4,096 tokens, but check.
* Rebalanced and size-matched training sets, format-swapped test sets, RAGognize in this format.
