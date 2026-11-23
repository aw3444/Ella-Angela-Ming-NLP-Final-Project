import torch
import torch.nn.functional as F

# product-of-experts loss (Clark et al., 2019), training only; at inference the detector runs alone
# with b = task-only P(token hallucinated | task), for each token with label != -100:
#   loss = -log_softmax(log_softmax(logits) + [log(1 - b), log(b)])[label], averaged over tokens
# logits: [batch, seq_len, 2]; labels: [batch, seq_len], 0 = supported, 1 = hallucinated, -100 = ignored
# bias_probabilities: [batch] (one task rate per answer) or [batch, seq_len]


def poe_loss(logits, labels, bias_probabilities, epsilon=1e-6):
    mask = labels != -100
    if bias_probabilities.dim() == 1:
        bias_probabilities = bias_probabilities[:, None].expand_as(labels)

    # fixed bias expert: no gradient, float32 (1 - epsilon rounds to 1 in bfloat16), neutral 0.5 at
    # ignored tokens so padding can't cause NaN gradients, and clipped so neither log is -inf
    bias_probabilities = bias_probabilities.detach().to(logits.device, torch.float32)
    bias_probabilities = torch.where(mask, bias_probabilities, 0.5).clamp(epsilon, 1 - epsilon)
    bias_log_probs = torch.stack([torch.log1p(-bias_probabilities), torch.log(bias_probabilities)], dim=-1)

    # product of experts: add log-probabilities, then renormalize over the two classes
    main_log_probs = F.log_softmax(logits.float(), dim=-1)
    combined_log_probs = F.log_softmax(main_log_probs + bias_log_probs, dim=-1)

    token_losses = F.nll_loss(
        combined_log_probs.reshape(-1, 2),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    # mean over labeled tokens, like the model's built-in loss; 0 instead of NaN if all are ignored
    return token_losses.sum() / mask.sum().clamp(min=1)
