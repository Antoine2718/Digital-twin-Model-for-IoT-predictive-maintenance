import yaml
import torch
from src.data_loader import get_dataloaders
from src.models.digital_twin import DigitalTwin
from src.feature_engineering import normalize_batch, add_statistical_channels

def test_forward_shapes():
    _, _, _, cfg = get_dataloaders("./config.yaml")
    loader, _, _, _ = get_dataloaders("./config.yaml")
    batch = next(iter(loader))
    x = batch["x"]
    x = normalize_batch(x)
    x = add_statistical_channels(x)
    input_dim = x.shape[-1]
    model = DigitalTwin(input_dim=input_dim, cfg=cfg)
    out = model(x)
    assert out["rul"].shape[0] == x.shape[0]
    assert out["fail_logit"].shape == (x.shape[0], 1)
    assert out["health_seq"].shape[1] == x.shape[1]
