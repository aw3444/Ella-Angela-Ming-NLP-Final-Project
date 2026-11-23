import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from data.evaluate import evaluate
from src.models.poe_loss import poe_loss
from src.models.task_only import TaskOnlyBaseline


def make_samples():
    # same fields as data/processed/ragtruth_data.json
    return [
        {"id": "q1", "task_type": "QA", "prompt": "Hours are 11am to 10pm",
         "answer": "Joe is open 24 hours", "labels": [{"start": 12, "end": 20}]},  # "24 hours": 2 of 5 words
        {"id": "q2", "task_type": "QA", "prompt": "Paris is the capital",
         "answer": "The capital is Paris", "labels": []},  # 0 of 4 words
        {"id": "d1", "task_type": "Data2txt", "prompt": "Opens at noon",
         "answer": "It opens at noon", "labels": [{"start": 0, "end": 8}]},  # "It opens": 2 of 4 words
    ]


def make_word_tokenizer():
    # one token per word with ModernBERT's input layout, so counts are easy to check offline
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")


def test_response_rates_and_shared_evaluation():
    samples = make_samples()
    model = TaskOnlyBaseline().fit(samples)

    # QA: 1 of 2 answers hallucinated; Data2txt: 1 of 1
    assert model.response_rates == pytest.approx({"QA": 0.5, "Data2txt": 1.0})
    assert model.token_rates == {}  # no tokenizer given

    predictions = model.predict(samples)
    assert predictions["q1"] == {"response_score": 0.5}

    results = evaluate(samples, predictions, threshold=0.6)  # flags the Data2txt answer only
    assert results["ALL"]["response"]["precision"] == pytest.approx(1.0)
    assert results["ALL"]["response"]["recall"] == pytest.approx(0.5)
    assert results["ALL"]["span"] is None  # answer-level model: no span scores


def test_token_rates_for_poe_bias():
    model = TaskOnlyBaseline().fit(make_samples(), make_word_tokenizer())

    # QA: 2 of 9 answer tokens; Data2txt: 2 of 4
    assert model.token_rates == pytest.approx({"QA": 2 / 9, "Data2txt": 0.5})
    with pytest.raises(KeyError):
        model.token_probability("Summary")  # unseen task

    bias_probabilities = torch.tensor([model.token_probability(task) for task in ["QA", "Data2txt"]])
    loss = poe_loss(torch.zeros(2, 3, 2), torch.tensor([[-100, 0, 1], [-100, 1, 0]]), bias_probabilities)
    assert torch.isfinite(loss)
