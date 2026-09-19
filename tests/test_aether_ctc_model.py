import torch

from aether_v3.config import AetherSpeechConfig, CTCConfig
from aether_v3.models.aether_ctc_model import AetherCTCModel
from aether_v3.models.ctc_head import compute_ctc_loss


def test_end_to_end_forward_and_backward():
    speech_cfg = AetherSpeechConfig(
        semantic_vocab_size=16, hidden_size=8, num_layers=2, num_heads=2, ffn_size=16, dropout=0.0
    )
    ctc_cfg = CTCConfig(vocab_size=6, blank_id=5)
    model = AetherCTCModel(speech_cfg, ctc_cfg)

    codes = torch.randint(0, speech_cfg.semantic_vocab_size, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.bool)
    log_probs = model(codes, mask)
    assert log_probs.shape == (2, 12, ctc_cfg.vocab_size)

    targets = torch.tensor([0, 1, 2, 3])
    input_lengths = torch.tensor([12, 12])
    target_lengths = torch.tensor([2, 2])
    loss = compute_ctc_loss(log_probs, targets, input_lengths, target_lengths, ctc_cfg.blank_id)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)
