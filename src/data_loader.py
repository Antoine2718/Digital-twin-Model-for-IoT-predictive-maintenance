# src/data_loader.py
import os
import math
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class IoTSensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y_rul: np.ndarray, y_fail: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)          # [N, T, F]
        self.y_rul = torch.tensor(y_rul, dtype=torch.float32)  # [N, 1]
        self.y_fail = torch.tensor(y_fail, dtype=torch.float32) # [N, 1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y_rul": self.y_rul[idx],
            "y_fail": self.y_fail[idx],
        }

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def _simulate_asset_series(num_sensors: int, max_cycles: int) -> pd.DataFrame:
    # Simule des capteurs avec dérive, bruit, dérivation par environnement
    t = np.arange(max_cycles)
    env = np.clip(np.sin(t / 200) + 0.2 * np.cos(t / 50), -1.5, 1.5)
    data = {}
    for s in range(num_sensors):
        drift = 0.0005 * t + 0.02 * np.random.randn(max_cycles)
        vib = 0.4 * np.sin(t / (30 + s)) + 0.1 * np.random.randn(max_cycles)
        temp = 0.6 * (env + 0.3 * np.random.randn(max_cycles))
        series = 0.2 * drift + vib + 0.1 * temp
        data[f"sensor_{s}"] = series
    df = pd.DataFrame(data)
    df["env"] = env
    return df

def _compute_rul_and_failure(df: pd.DataFrame, failure_threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    # Heuristique: un indice de santé synthétique basé sur variance et amplitude
    health_idx = 1.0 - np.clip((df.filter(regex="sensor_").abs().mean(axis=1) / 5.0), 0, 1)
    failure_signal = (1.0 - health_idx)  # augmente avec la dégradation
    failure = (failure_signal > failure_threshold).astype(np.float32)
    # RUL synthétique: cycles restants jusqu’à atteindre seuil
    # Si jamais non atteint, on met une grande valeur
    rul = []
    fs = failure_signal.values
    for i in range(len(fs)):
        horizon = np.argmax(fs[i:] > failure_threshold)
        if horizon == 0 and fs[i] <= failure_threshold:
            # pas trouvé
            rul.append(1000.0)
        else:
            # si trouvé, horizon est index relatif
            rul.append(float(horizon))
    return np.array(rul).reshape(-1, 1), failure.reshape(-1, 1)

def make_windows(df: pd.DataFrame, window: int, step: int, num_sensors: int):
    X, y_rul, y_fail = [], [], []
    df_sensors = df[[f"sensor_{i}" for i in range(num_sensors)]].values
    env = df["env"].values
    rul, fail = _compute_rul_and_failure(df)
    for start in range(0, len(df) - window, step):
        end = start + window
        seg = df_sensors[start:end]  # [T, F]
        # concat env en feature additionnelle
        env_seg = env[start:end].reshape(-1, 1)
        X.append(np.concatenate([seg, env_seg], axis=1))
        y_rul.append(rul[end-1])   # label au dernier pas
        y_fail.append(fail[end-1])
    return np.array(X), np.array(y_rul), np.array(y_fail)

def build_synthetic_dataset(cfg: Dict):
    num_assets = cfg["data"]["synthetic"]["num_assets"]
    max_cycles = cfg["data"]["synthetic"]["max_cycles"]
    num_sensors = cfg["data"]["num_sensors"]
    window = cfg["data"]["window_size"]
    step = cfg["data"]["step_size"]

    X_all, y_rul_all, y_fail_all = [], [], []
    for _ in range(num_assets):
        df = _simulate_asset_series(num_sensors, max_cycles)
        X, y_rul, y_fail = make_windows(df, window, step, num_sensors)
        X_all.append(X); y_rul_all.append(y_rul); y_fail_all.append(y_fail)

    X = np.vstack(X_all)
    y_rul = np.vstack(y_rul_all)
    y_fail = np.vstack(y_fail_all)
    return X, y_rul, y_fail

def get_dataloaders(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    root = Path(cfg["data"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    processed_dir = root / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Pour démonstration, on génère des données synthétiques
    X, y_rul, y_fail = build_synthetic_dataset(cfg)

    dataset = IoTSensorDataset(X, y_rul, y_fail)
    n = len(dataset)
    train_len = int(n * cfg["data"]["train_ratio"])
    val_len = int(n * cfg["data"]["val_ratio"])
    test_len = n - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    batch_size = cfg["training"]["batch_size"]
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
        cfg,
    )
