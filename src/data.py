"""Deterministic datasets and leakage-safe partitions."""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from sklearn.datasets import load_digits


def cnn_data(seed=42):
    digits = load_digits()
    labels = digits.target.astype(np.int64)
    groups = np.empty(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        groups[indices] = np.arange(len(indices)) % 12

    images = np.empty((len(labels), 32, 32, 1), dtype=np.float32)
    settings = np.random.default_rng(seed + 1).uniform(size=(12, 3))
    for i, original in enumerate(digits.images):
        group = groups[i]
        resized = zoom(original / 16.0, 4, order=1)
        contrast = 0.82 + 0.35 * settings[group, 0]
        brightness = -0.06 + 0.12 * settings[group, 1]
        blur = 0.15 + 0.38 * settings[group, 2]
        acquired = gaussian_filter(resized * contrast + brightness, blur)
        images[i, :, :, 0] = np.clip(acquired + rng.normal(0, 0.025, (32, 32)), 0, 1)

    development = np.flatnonzero(groups < 8)
    test = np.flatnonzero(groups >= 8)
    assert not set(groups[development]) & set(groups[test])
    return images, labels, groups, development, test


def temporal_series(length=1600, seed=42):
    rng = np.random.default_rng(seed)
    target = np.zeros(length, dtype=np.float32)
    time = np.arange(length, dtype=np.float32)
    sine = np.sin(2 * np.pi * time / 48)
    cosine = np.cos(2 * np.pi * time / 48)
    control = np.zeros(length, dtype=np.float32)
    expected = np.zeros(length, dtype=np.float32)
    noise = rng.normal(0, 0.11, length)
    control_noise = rng.normal(0, 0.14, length)
    for t in range(1, length):
        control[t] = 0.78 * control[t - 1] + control_noise[t]
        mean = 0.65 * target[t - 1] + 0.25 * sine[t] + 0.13 * control[t - 1]
        expected[t] = mean
        target[t] = mean + noise[t]
    features = np.column_stack([target, sine, cosine, control]).astype(np.float32)
    return features, target, expected


def chronological_windows(features, target, start, stop, lookback=36):
    prediction_times = np.arange(start + lookback, stop)
    windows = np.stack([features[t - lookback : t] for t in prediction_times])
    assert prediction_times[0] - lookback >= start
    assert prediction_times[-1] < stop
    return windows.astype(np.float32), target[prediction_times], prediction_times


def temporal_splits(lookback=36):
    features, target, _ = temporal_series()
    ranges = {"train": (0, 960), "validation": (960, 1280), "test": (1280, 1600)}
    splits = {
        name: chronological_windows(features, target, start, stop, lookback)
        for name, (start, stop) in ranges.items()
    }
    train_x, train_y, _ = splits["train"]
    feature_mean = train_x.reshape(-1, 4).mean(axis=0)
    feature_std = train_x.reshape(-1, 4).std(axis=0) + 1e-8
    target_mean = train_y.mean()
    target_std = train_y.std() + 1e-8
    scaled = {
        name: ((x - feature_mean) / feature_std, (y - target_mean) / target_std, times)
        for name, (x, y, times) in splits.items()
    }
    scalers = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": np.array(target_mean),
        "target_std": np.array(target_std),
    }
    return scaled, scalers
