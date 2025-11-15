import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, AUROC, Accuracy

class MultiTaskLoss(nn.Module):
    def __init__(self, w_rul=1.0, w_fail=1.0):
        super().__init__()
        self.w_rul = w_rul
        self.w_fail = w_fail
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        loss_rul = self.mse(preds["rul"], targets["y_rul"])
        loss_fail = self.bce(preds["fail_logit"], targets["y_fail"])
        return self.w_rul * loss_rul + self.w_fail * loss_fail, {"loss_rul": loss_rul.item(), "loss_fail": loss_fail.item()}

class Metrics:
    def __init__(self):
        self.mse = MeanSquaredError()
        self.acc = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")

    def update(self, preds, targets):
        self.mse.update(preds["rul"].detach(), targets["y_rul"].detach())
        probs = torch.sigmoid(preds["fail_logit"].detach())
        self.acc.update(probs, targets["y_fail"].detach())
        self.auroc.update(probs, targets["y_fail"].detach())

    def compute(self):
        return {
            "mse_rul": float(self.mse.compute().item()),
            "acc_fail": float(self.acc.compute().item()),
            "auroc_fail": float(self.auroc.compute().item())
        }

    def reset(self):
        self.mse.reset(); self.acc.reset(); self.auroc.reset()
