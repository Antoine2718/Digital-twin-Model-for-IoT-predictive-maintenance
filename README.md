# Model for IoT predictive maintenance
### This model includes IoT sequential data loading, feature engineering, a hybrid digital twin (learning + simplified physical model), multitasking training (RUL regression + breakdown classification), evaluation and inference. The project includes a synthetic data generator to test without proprietary dataset.

This project implements a hybrid digital twin combining:

- A sequential sensor encoder (Transformer or LSTM)

- A simplified physical degradation model

- A merger to predict the RUL (regression) and the probability of failure (classification)

## Install

```bash

Python3 -m venv .venv

Source .venv/bin/activate

Pip install -r requirements.txt
```
## Train 
```bash
python -m src.train
```
## Evaluate
```bash
python -m src.evaluate
```
## Interference
```python
from src.inference import TwinPredictor
import numpy as np

predictor = TwinPredictor("./config.yaml")
# Exemple: T=256, F = num_sensors + 1 (env)
x_np = np.random.randn(256, 13)  # 12 capteurs + 1 env
res = predictor.predict(x_np)
print(res)
```
## Notes

Real data must be placed in data/raw/ and adapted in data_loader.py.

The physical model is configurable via config.yaml (alpha/beta).

The pipeline adds statistical channels for robustness and calibration.
