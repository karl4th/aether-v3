import torch

from aether_v3.config import BridgeConfig, ConnectorConfig, ResamplerConfig
from aether_v3.models.bridge import AetherBridge
from aether_v3.models.connector import AetherConnector


def test_bridge_shape_and_backward():
    cfg = BridgeConfig(input_dim=8, intermediate_dim=16, output_dim=20, dropout=0.0)
    bridge = AetherBridge(cfg)
    x = torch.randn(2, 5, 8, requires_grad=True)
    out = bridge(x)
    assert out.shape == (2, 5, 20)
    out.sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)


def test_connector_ratio_one_preserves_length():
    cfg = ConnectorConfig(
        resampler=ResamplerConfig(enabled=False, ratio=1),
        bridge=BridgeConfig(input_dim=8, intermediate_dim=16, output_dim=20, dropout=0.0),
    )
    connector = AetherConnector(speech_hidden_size=8, cfg=cfg)
    x = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.bool)
    embeds, out_mask = connector(x, mask)
    assert embeds.shape == (2, 10, 20)
    assert out_mask is mask


def test_connector_ratio_four_downsamples():
    cfg = ConnectorConfig(
        resampler=ResamplerConfig(enabled=True, ratio=4, num_heads=2, ffn_size=16, dropout=0.0),
        bridge=BridgeConfig(input_dim=8, intermediate_dim=16, output_dim=20, dropout=0.0),
    )
    connector = AetherConnector(speech_hidden_size=8, cfg=cfg)
    x = torch.randn(2, 16, 8)
    mask = torch.ones(2, 16, dtype=torch.bool)
    embeds, out_mask = connector(x, mask)
    assert embeds.shape[0] == 2
    assert embeds.shape[1] < 16
    assert embeds.shape[2] == 20
    assert out_mask.shape[:2] == embeds.shape[:2]


def test_connector_boundary_embeddings_are_learnable_params():
    cfg = ConnectorConfig(
        resampler=ResamplerConfig(enabled=False, ratio=1),
        bridge=BridgeConfig(input_dim=8, intermediate_dim=16, output_dim=20, dropout=0.0),
    )
    connector = AetherConnector(speech_hidden_size=8, cfg=cfg)
    assert connector.speech_start.shape == (20,)
    assert connector.speech_end.shape == (20,)
    assert connector.speech_start.requires_grad
    assert connector.speech_end.requires_grad


def test_bridge_output_scale_stays_fp32_when_connector_is_cast():
    connector = AetherConnector(
        12,
        ConnectorConfig(
            resampler=ResamplerConfig(enabled=False, ratio=1),
            bridge=BridgeConfig(input_dim=12, intermediate_dim=24, output_dim=20),
        ),
    )

    connector.to(dtype=torch.bfloat16)

    assert connector.bridge.output_scale.dtype == torch.float32
    assert connector.bridge.mlp.in_proj.weight.dtype == torch.bfloat16
