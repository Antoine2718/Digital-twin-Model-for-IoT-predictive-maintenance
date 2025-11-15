import os
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .data_loader import get_dataloaders
from .feature_engineering import normalize_batch, add_statistical_channels
from .models.digital_twin import DigitalTwin
from .losses import MultiTaskLoss, Metrics

def get_optimizer(params, cfg):
    opt_name = cfg["training"]["optimizer"].lower()
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    if opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    return Adam(params, lr=lr, weight_decay=wd)

def train(cfg_path: str = "./config.yaml"):
    train_loader, val_loader, test_loader, cfg = get_dataloaders(cfg_path)

    # dimensions: dernière feature = env
    input_dim = next(iter(train_loader))["x"].shape[-1]
    model = DigitalTwin(input_dim=input_dim, cfg=cfg).to(cfg["training"]["device"])
    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"]) if cfg["training"]["scheduler"] == "cosine" else None
    criterion = MultiTaskLoss(w_rul=1.0, w_fail=1.0)
    metrics = Metrics()

    best_val = float("inf")
    patience = cfg["training"]["patience"]
    wait = 0
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        metrics.reset()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x = batch["x"].to(cfg["training"]["device"])
            x = normalize_batch(x)
            x = add_statistical_channels(x)

            y_rul = batch["y_rul"].to(cfg["training"]["device"])
            y_fail = batch["y_fail"].to(cfg["training"]["device"])

            preds = model(x)
            loss, parts = criterion(preds, {"y_rul": y_rul, "y_fail": y_fail})
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()
            running_loss += loss.item()
            metrics.update(preds, {"y_rul": y_rul, "y_fail": y_fail})

        train_stats = metrics.compute()
        metrics.reset()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x = batch["x"].to(cfg["training"]["device"])
                x = normalize_batch(x)
                x = add_statistical_channels(x)

                y_rul = batch["y_rul"].to(cfg["training"]["device"])
                y_fail = batch["y_fail"].to(cfg["training"]["device"])
                preds = model(x)
                loss, parts = criterion(preds, {"y_rul": y_rul, "y_fail": y_fail})
                val_loss += loss.item()
                metrics.update(preds, {"y_rul": y_rul, "y_fail": y_fail})
        val_stats = metrics.compute()
        metrics.reset()

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch}: train_loss={running_loss/len(train_loader):.4f} | val_loss={val_loss/len(val_loader):.4f} | val_mse_rul={val_stats['mse_rul']:.4f} | val_auroc_fail={val_stats['auroc_fail']:.4f}")

        # Early stopping
        monitor = val_loss / len(val_loader)
        if monitor < best_val:
            best_val = monitor
            wait = 0
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # Plot exemple: santé estimée sur un batch test
    plots_dir = Path(cfg["logging"]["plots_dir"]); plots_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(test_loader))
    x = batch["x"].to(cfg["training"]["device"])
    x = normalize_batch(x)
    x = add_statistical_channels(x)
    with torch.no_grad():
        out = model(x)
    health = out["health_seq"][0].cpu().numpy().squeeze()
    plt.figure(figsize=(8,3))
    plt.plot(health)
    plt.title("Séquence de santé estimée (jumeau numérique)")
    plt.xlabel("Temps"); plt.ylabel("Santé")
    plt.tight_layout()
    plt.savefig(plots_dir / "health_example.png")
    print("Training complete. Best checkpoint saved.")

if __name__ == "__main__":
    train()
