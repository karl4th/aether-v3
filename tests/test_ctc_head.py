import torch

from aether_v3.models.ctc_head import CTCHead, compute_ctc_loss


def test_ctc_head_output_is_log_probabilities():
    head = CTCHead(hidden_size=8, vocab_size=5)
    x = torch.randn(2, 4, 8)
    log_probs = head(x)
    assert log_probs.shape == (2, 4, 5)
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)


def test_compute_ctc_loss_is_finite_for_feasible_alignment():
    torch.manual_seed(0)
    log_probs = torch.log_softmax(torch.randn(2, 10, 5), dim=-1)
    targets = torch.tensor([1, 2, 3, 1, 2])
    input_lengths = torch.tensor([10, 10])
    target_lengths = torch.tensor([3, 2])
    loss = compute_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank_id=4)
    assert torch.isfinite(loss)


def test_compute_ctc_loss_zero_infinity_for_infeasible_alignment():
    """Target longer than input -> CTC alignment is infeasible;
    zero_infinity=True must clamp that example's loss/gradient instead of
    returning inf/nan (this is exactly the case `_ctc_min_input_length` in
    mimi_cache.py is used to detect and filter out of the training cache)."""
    torch.manual_seed(0)
    log_probs = torch.log_softmax(torch.randn(1, 2, 5), dim=-1)
    targets = torch.tensor([1, 1, 1, 1])  # 4 labels, all adjacent-equal
    input_lengths = torch.tensor([2])
    target_lengths = torch.tensor([4])
    loss = compute_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank_id=4)
    assert torch.isfinite(loss)
