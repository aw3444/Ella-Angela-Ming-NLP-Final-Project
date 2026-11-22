"""Shared encoding function: one example -> model inputs + token labels.

Everyone should use `encode_example` so every detector sees identical labels.

Layout (same as LettuceDetect):  [CLS] prompt [SEP] answer [SEP]
  * prompt tokens and special tokens get label -100 (ignored by the loss)
  * an answer token gets 1 if it overlaps any hallucinated span, else 0
  * if too long, only the prompt is cut (truncation="only_first"); the answer is never cut

answer_only=True gives the input for the answer-only baseline:  [CLS] answer [SEP]

Usage:
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
  enc = encode_example(tok, sample["prompt"], sample["answer"], sample["labels"])
"""

MAX_LENGTH = 4096  # same as LettuceDetect


def _token_label(tok_start: int, tok_end: int, spans: list[dict]) -> int:
    for s in spans:
        if tok_end > s["start"] and tok_start < s["end"]:
            return 1
    return 0


def encode_example(tokenizer, prompt: str, answer: str, spans: list[dict],
                   max_length: int = MAX_LENGTH, answer_only: bool = False) -> dict:
    """Return {"input_ids", "attention_mask", "labels", "answer_offsets", "truncated"}.

    answer_offsets[i] is the (start, end) character span in `answer` for token i,
    or None for non-answer tokens; the evaluation code uses it to turn token
    predictions back into character spans.
    """
    if answer_only:
        enc = tokenizer(answer, truncation=True, max_length=max_length,
                        return_offsets_mapping=True, add_special_tokens=True)
        answer_seq_id = 0
        truncated = False
    else:
        full_len = len(tokenizer(prompt, answer, add_special_tokens=True)["input_ids"])
        enc = tokenizer(prompt, answer, truncation="only_first", max_length=max_length,
                        return_offsets_mapping=True, add_special_tokens=True)
        answer_seq_id = 1
        truncated = full_len > max_length

    seq_ids = enc.sequence_ids()
    labels, answer_offsets = [], []
    for sid, (start, end) in zip(seq_ids, enc["offset_mapping"]):
        if sid == answer_seq_id:
            # offsets of the second sequence are relative to `answer` itself
            labels.append(_token_label(start, end, spans))
            answer_offsets.append((start, end))
        else:
            labels.append(-100)
            answer_offsets.append(None)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "answer_offsets": answer_offsets,
        "truncated": truncated,
    }
