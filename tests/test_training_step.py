import torch
from src.losses import MultiTaskLoss
from src.models.digital_twin import DigitalTwin
from src.data_loader import get_dataloaders
from src.feature_engineering import normalize_batch, add_statistical_channels

def test_training_step():
    train_loader, _, _, cfg = get_dataloaders("./config.yaml")
    batch = next(iter(train_loader))
    x = batch["x"]
    x = normalize_batch(x)
    x = add_statistical_channels(x)
    input_dim = x.shape[-1]
    model = DigitalTwin(input_dim=input_dim, cfg=cfg)
    preds = model(x)
    criterion = MultiTaskLoss()
    loss, parts = criterion(preds, {"y_rul": batch["y_rul"], "y_fail": batch["y_fail"]})
    assert loss.item() > 0.0
