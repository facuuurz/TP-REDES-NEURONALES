"""Load exported models and perform one example inference per task."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import numpy as np
from tensorflow import keras

from src.data import cnn_data, temporal_splits


def main():
    root = Path("artifacts")
    images, labels, _, _, held_out = cnn_data()
    cnn = keras.models.load_model(root / "cnn.keras", compile=False)
    idx = held_out[0]
    digit = int(np.argmax(cnn.predict(images[idx : idx + 1], verbose=0)))
    print(f"CNN sample {idx}: predicted={digit}, actual={labels[idx]}")

    splits, _ = temporal_splits()
    features, targets, times = splits["test"]
    with np.load(root / "gru_scalers.npz") as scalers:
        gru = keras.models.load_model(root / "gru.keras", compile=False)
        forecast = float(gru.predict(features[:1], verbose=0)[0, 0] * scalers["target_std"] + scalers["target_mean"])
        actual = float(targets[0] * scalers["target_std"] + scalers["target_mean"])
    print(f"GRU time {times[0]}: predicted={forecast:.4f}, actual={actual:.4f}")


if __name__ == "__main__":
    main()
