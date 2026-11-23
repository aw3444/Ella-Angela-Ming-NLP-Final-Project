import math

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, ModernBertConfig

from src.models.poe_loss import poe_loss


def make_batch():
    generator = torch.Generator().manual_seed(4705)
    logits = torch.randn(2, 4, 2, generator=generator, requires_grad=True)
    labels = torch.tensor([
        [-100, 0, 1, 0],
        [1, 1, -100, -100],
    ])
    bias_probabilities = torch.tensor([0.3, 0.7])  # one task rate per answer
    return logits, labels, bias_probabilities


def test_poe_loss_hand_calculation():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    labels = torch.tensor([[0, 1]])

    loss = poe_loss(logits, labels, torch.tensor([0.8]))

    # token 0: p_poe(0) = 0.8808 * 0.2 / (0.8808 * 0.2 + 0.1192 * 0.8) = 0.6488
    # token 1: p_poe(1) = 0.7311 * 0.8 / (0.7311 * 0.8 + 0.2689 * 0.2) = 0.9158
    assert loss.shape == torch.Size([])
    assert loss.item() == pytest.approx((-math.log(0.6488) - math.log(0.9158)) / 2, abs=1e-4)


def test_uninformative_bias_matches_standard_detector_loss():
    # tiny random ModernBERT with the standard detector's head, so no download is needed
    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        pad_token_id=0,
        num_labels=2,
    )
    model = AutoModelForTokenClassification.from_config(config)

    input_ids = torch.randint(1, 100, (2, 6))
    labels = torch.tensor([
        [-100, -100, 0, 1, 1, -100],  # prompt, answer, [SEP]
        [-100, 0, 0, 0, -100, -100],
    ])
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)

    # with b = 0.5 the bias adds nothing, so PoE equals the standard detector's loss
    loss = poe_loss(outputs.logits, labels, torch.tensor([0.5, 0.5]))
    assert outputs.logits.shape == (2, 6, 2)
    assert loss.item() == pytest.approx(outputs.loss.item(), rel=1e-5)


def test_gradient_flows_only_through_the_detector():
    logits, labels, _ = make_batch()
    bias_probabilities = torch.tensor([0.3, 0.7], requires_grad=True)

    poe_loss(logits, labels, bias_probabilities).backward()

    # expected gradient: (p_poe - one_hot(label)) / number of labeled tokens, 0 at ignored tokens
    mask = labels != -100
    bias_log_probs = torch.stack([torch.log(1 - bias_probabilities), torch.log(bias_probabilities)], dim=-1)
    poe_probs = torch.softmax(logits.detach() + bias_log_probs.detach()[:, None, :], dim=-1)
    expected_gradient = (poe_probs - F.one_hot(labels.clamp(min=0), 2)) / mask.sum()
    expected_gradient[~mask] = 0.0

    torch.testing.assert_close(logits.grad, expected_gradient)
    assert bias_probabilities.grad is None  # the bias expert is fixed


def test_ignored_tokens():
    logits, labels, bias_probabilities = make_batch()
    ignored = labels == -100
    loss = poe_loss(logits, labels, bias_probabilities)

    # ignored logits don't change the loss
    changed_logits = logits.detach().clone()
    changed_logits[ignored] = torch.tensor([50.0, -50.0])
    assert poe_loss(changed_logits, labels, bias_probabilities).item() == pytest.approx(loss.item())

    # NaN bias at padded positions: same loss, finite gradients
    per_token_bias = bias_probabilities[:, None].expand(2, 4).clone()
    per_token_bias[ignored] = float("nan")
    per_token_loss = poe_loss(logits, labels, per_token_bias)
    per_token_loss.backward()
    assert per_token_loss.item() == pytest.approx(loss.item())
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[ignored] == 0)

    # all tokens ignored: 0 instead of NaN, no gradient
    empty_logits = torch.randn(2, 4, 2, requires_grad=True)
    empty_loss = poe_loss(empty_logits, torch.full((2, 4), -100), bias_probabilities)
    empty_loss.backward()
    assert empty_loss.item() == 0.0
    assert torch.all(empty_logits.grad == 0)


def test_extreme_bias_and_bfloat16_stay_finite():
    logits, labels, _ = make_batch()
    for bias_value in [0.0, 1.0]:
        loss = poe_loss(logits, labels, torch.tensor([bias_value, bias_value]))
        gradient, = torch.autograd.grad(loss, logits)
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()

    # 1 - 1e-6 rounds to 1.0 in bfloat16
    bfloat16_logits = torch.randn(1, 2, 2).to(torch.bfloat16).requires_grad_()
    bias_probabilities = torch.tensor([1 - 1e-6], dtype=torch.bfloat16)
    loss = poe_loss(bfloat16_logits, torch.tensor([[0, 1]]), bias_probabilities)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.isfinite(bfloat16_logits.grad.float()).all()
