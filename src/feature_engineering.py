import torch

def normalize_batch(x: torch.Tensor):
    # x: [B, T, F]
    mean = x.mean(dim=(0,1), keepdim=True)
    std = x.std(dim=(0,1), keepdim=True) + 1e-6
    return (x - mean) / std

def add_statistical_channels(x: torch.Tensor):
    # Ajoute des canaux statistiques pour robustesse
    # returns [B, T, F + 3]
    rolling_mean = x.mean(dim=1, keepdim=True).repeat(1, x.size(1), 1)
    rolling_std = x.std(dim=1, keepdim=True).repeat(1, x.size(1), 1)
    max_min = (x.max(dim=1, keepdim=True).values - x.min(dim=1, keepdim=True).values).repeat(1, x.size(1), 1)
    return torch.cat([x, rolling_mean, rolling_std, max_min], dim=-1)
