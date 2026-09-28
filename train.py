"""Train, evaluate, and export both assignment models."""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from tensorflow import keras

from src.data import cnn_data, temporal_splits
from src.models import build_cnn, build_gru


OUTPUT = Path("artifacts")


def setup():
    OUTPUT.mkdir(exist_ok=True)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def save_json(name, data):
    (OUTPUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def plot_history(history, name):
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name.lower()}_learning_curve.png", dpi=150)
    plt.close(fig)


def plot_cnn_folds(folds):
    positions = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.bar(positions - 0.18, [fold["training_accuracy"] for fold in folds], 0.36, label="Training")
    ax.bar(positions + 0.18, [fold["validation_accuracy"] for fold in folds], 0.36, label="Validation")
    ax.set(xticks=positions, xticklabels=[str(fold["fold"]) for fold in folds], xlabel="GroupKFold split", ylabel="Accuracy", ylim=(0.8, 1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "cnn_fold_accuracy.png", dpi=150)
    plt.close(fig)


def train_cnn():
    x, y, groups, development, test = cnn_data()
    folds = []
    chosen_epochs = []
    for fold, (train_pos, val_pos) in enumerate(
        GroupKFold(n_splits=4).split(x[development], y[development], groups[development]), 1
    ):
        train_idx, val_idx = development[train_pos], development[val_pos]
        assert not set(groups[train_idx]) & set(groups[val_idx])
        keras.backend.clear_session()
        keras.utils.set_random_seed(42 + fold)
        model = build_cnn()
        history = model.fit(
            x[train_idx], y[train_idx],
            validation_data=(x[val_idx], y[val_idx]),
            epochs=30, batch_size=32, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        )
        prediction = np.argmax(model.predict(x[val_idx], verbose=0), axis=1)
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        chosen_epochs.append(best_epoch)
        folds.append({
            "fold": fold,
            "training_groups": sorted(set(groups[train_idx].tolist())),
            "validation_groups": sorted(set(groups[val_idx].tolist())),
            "training_accuracy": round(float(model.evaluate(x[train_idx], y[train_idx], verbose=0)[1]), 4),
            "validation_accuracy": round(float(accuracy_score(y[val_idx], prediction)), 4),
            "best_epoch": best_epoch,
        })
        print(f"CNN fold {fold}: {folds[-1]}", flush=True)

    epochs = int(np.median(chosen_epochs))
    keras.backend.clear_session()
    keras.utils.set_random_seed(142)
    model = build_cnn()
    history = model.fit(x[development], y[development], epochs=epochs, batch_size=32, verbose=0)
    model.save(OUTPUT / "cnn.keras")
    model.save_weights(OUTPUT / "cnn.weights.h5")
    prediction = np.argmax(model.predict(x[test], verbose=0), axis=1)
    metrics = {
        "dataset": "sklearn digits with deterministic simulated visual batches",
        "samples": {"development": len(development), "test": len(test)},
        "test_groups": sorted(set(groups[test].tolist())),
        "group_kfold": folds,
        "cross_validation_accuracy_mean": round(float(np.mean([f["validation_accuracy"] for f in folds])), 4),
        "cross_validation_accuracy_std": round(float(np.std([f["validation_accuracy"] for f in folds])), 4),
        "selected_epochs": epochs,
        "final_training_accuracy": round(float(model.evaluate(x[development], y[development], verbose=0)[1]), 4),
        "test_accuracy": round(float(accuracy_score(y[test], prediction)), 4),
        "test_macro_f1": round(float(f1_score(y[test], prediction, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y[test], prediction).tolist(),
    }
    save_json("cnn_metrics.json", metrics)
    plot_cnn_folds(folds)
    print("CNN test:", metrics["test_accuracy"], metrics["test_macro_f1"], flush=True)
    return metrics


def train_gru():
    splits, scalers = temporal_splits()
    np.savez(OUTPUT / "gru_scalers.npz", **scalers)
    train_x, train_y, train_times = splits["train"]
    val_x, val_y, val_times = splits["validation"]
    test_x, test_y, test_times = splits["test"]
    assert train_times.max() < val_times.min() < test_times.min()
    keras.backend.clear_session()
    keras.utils.set_random_seed(242)
    model = build_gru()
    history = model.fit(
        train_x, train_y, validation_data=(val_x, val_y),
        epochs=55, batch_size=32, shuffle=False, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)],
    )
    model.save(OUTPUT / "gru.keras")
    model.save_weights(OUTPUT / "gru.weights.h5")
    plot_history(history, "GRU")
    mean, std = float(scalers["target_mean"]), float(scalers["target_std"])
    actual = test_y * std + mean
    prediction = model.predict(test_x, verbose=0).ravel() * std + mean
    persistence = test_x[:, -1, 0] * scalers["feature_std"][0] + scalers["feature_mean"][0]
    residuals = np.abs(actual - prediction)
    metrics = {
        "dataset": "deterministic synthetic multivariate autoregressive time series",
        "ranges": {"train": [0, 960], "validation": [960, 1280], "test": [1280, 1600]},
        "window_counts": {name: len(part[0]) for name, part in splits.items()},
        "lookback": train_x.shape[1],
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "training_mae": round(float(mean_absolute_error(train_y * std + mean, model.predict(train_x, verbose=0).ravel() * std + mean)), 4),
        "validation_mae": round(float(mean_absolute_error(val_y * std + mean, model.predict(val_x, verbose=0).ravel() * std + mean)), 4),
        "test_mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(actual, prediction))), 4),
        "persistence_test_mae": round(float(mean_absolute_error(actual, persistence)), 4),
        "persistence_test_rmse": round(float(np.sqrt(mean_squared_error(actual, persistence))), 4),
        "test_mae_by_chronological_quarter": [round(float(chunk.mean()), 4) for chunk in np.array_split(residuals, 4)],
    }
    save_json("gru_metrics.json", metrics)
    print("GRU test:", metrics["test_mae"], metrics["test_rmse"], flush=True)
    return metrics


if __name__ == "__main__":
    setup()
    train_cnn()
    train_gru()
