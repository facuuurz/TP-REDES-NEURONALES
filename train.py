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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from tensorflow import keras

from src.data import cnn_data, temporal_series, temporal_splits
from src.models import build_cnn, build_recurrent


OUTPUT = Path("artifacts")
CNN_CONFIGS = {
    "final": {"last_stride": 2, "epochs": 150, "patience": 12},
    "truncated_budget": {"last_stride": 2, "epochs": 30, "patience": 5},
    "stride_1": {"last_stride": 1, "epochs": 150, "patience": 12},
}
RECURRENT_CELLS = ["gru", "lstm", "simple_rnn"]


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


def plot_cnn_learning_curves(folds):
    fig, axes = plt.subplots(1, len(folds), figsize=(10, 2.8), sharey=True)
    for ax, fold in zip(axes, folds):
        ax.plot(fold["history"]["loss"], label="Training")
        ax.plot(fold["history"]["val_loss"], label="Validation")
        ax.axvline(fold["best_epoch"] - 1, color="grey", linestyle=":", linewidth=1)
        ax.set(title=f"Fold {fold['fold']}", xlabel="Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "cnn_learning_curves.png", dpi=150)
    plt.close(fig)


def plot_cnn_folds(results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    titles = {"truncated_budget": "30 epochs, patience 5", "final": "150 epochs, patience 12"}
    for ax, (name, title) in zip(axes, titles.items()):
        folds = results[name]["folds"]
        positions = np.arange(len(folds))
        ax.bar(positions - 0.18, [fold["training_accuracy"] for fold in folds], 0.36, label="Training")
        ax.bar(positions + 0.18, [fold["validation_accuracy"] for fold in folds], 0.36, label="Validation")
        ax.set(xticks=positions, xticklabels=[str(fold["fold"]) for fold in folds], xlabel="GroupKFold split", title=title, ylim=(0.85, 1.0))
    axes[0].set_ylabel("Accuracy")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUTPUT / "cnn_fold_accuracy.png", dpi=150)
    plt.close(fig)


def group_cross_validation(x, y, groups, development, last_stride, epochs, patience):
    folds = []
    splits = GroupKFold(n_splits=4).split(x[development], y[development], groups[development])
    for fold, (train_pos, val_pos) in enumerate(splits, 1):
        train_idx, val_idx = development[train_pos], development[val_pos]
        assert not set(groups[train_idx]) & set(groups[val_idx])
        keras.backend.clear_session()
        keras.utils.set_random_seed(42 + fold)
        model = build_cnn(last_stride)
        history = model.fit(
            x[train_idx], y[train_idx],
            validation_data=(x[val_idx], y[val_idx]),
            epochs=epochs, batch_size=32, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
        )
        prediction = np.argmax(model.predict(x[val_idx], verbose=0), axis=1)
        folds.append({
            "fold": fold,
            "training_groups": sorted(set(groups[train_idx].tolist())),
            "validation_groups": sorted(set(groups[val_idx].tolist())),
            "epochs_run": len(history.history["loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "training_accuracy": round(float(model.evaluate(x[train_idx], y[train_idx], verbose=0)[1]), 4),
            "validation_accuracy": round(float(accuracy_score(y[val_idx], prediction)), 4),
            "history": {key: [round(float(v), 4) for v in values] for key, values in history.history.items() if "loss" in key},
        })
        print(f"CNN stride={last_stride} epochs={epochs} fold {fold}: {folds[-1]['validation_accuracy']}", flush=True)
    validation = [fold["validation_accuracy"] for fold in folds]
    training = [fold["training_accuracy"] for fold in folds]
    return {
        "configuration": {"last_stride": last_stride, "max_epochs": epochs, "patience": patience},
        "folds": folds,
        "training_accuracy_mean": round(float(np.mean(training)), 4),
        "validation_accuracy_mean": round(float(np.mean(validation)), 4),
        "validation_accuracy_std": round(float(np.std(validation)), 4),
        "generalization_gap_mean": round(float(np.mean(training) - np.mean(validation)), 4),
    }


def logistic_reference(x, y, groups, development, test):
    flat = x.reshape(len(x), -1)
    scores = []
    for train_pos, val_pos in GroupKFold(n_splits=4).split(flat[development], y[development], groups[development]):
        train_idx, val_idx = development[train_pos], development[val_pos]
        model = LogisticRegression(C=0.1, max_iter=3000).fit(flat[train_idx], y[train_idx])
        scores.append(model.score(flat[val_idx], y[val_idx]))
    model = LogisticRegression(C=0.1, max_iter=3000).fit(flat[development], y[development])
    return {
        "validation_accuracy_by_fold": [round(float(s), 4) for s in scores],
        "validation_accuracy_mean": round(float(np.mean(scores)), 4),
        "test_accuracy": round(float(model.score(flat[test], y[test])), 4),
    }


def layer_shapes(model):
    return [
        {"layer": layer.name, "output_shape": list(layer.output.shape[1:]), "parameters": layer.count_params()}
        for layer in model.layers
    ]


def train_cnn():
    x, y, groups, development, test = cnn_data()
    results = {
        name: group_cross_validation(x, y, groups, development, config["last_stride"], config["epochs"], config["patience"])
        for name, config in CNN_CONFIGS.items()
    }
    epochs = int(np.median([fold["best_epoch"] for fold in results["final"]["folds"]]))
    keras.backend.clear_session()
    keras.utils.set_random_seed(142)
    model = build_cnn(CNN_CONFIGS["final"]["last_stride"])
    model.fit(x[development], y[development], epochs=epochs, batch_size=32, verbose=0)
    model.save(OUTPUT / "cnn.keras")
    model.save_weights(OUTPUT / "cnn.weights.h5")
    prediction = np.argmax(model.predict(x[test], verbose=0), axis=1)
    metrics = {
        "dataset": "sklearn digits with deterministic simulated visual batches",
        "samples": {"development": len(development), "test": len(test)},
        "test_groups": sorted(set(groups[test].tolist())),
        "architecture": layer_shapes(model),
        "total_parameters": model.count_params(),
        "group_kfold": results,
        "logistic_regression_reference": logistic_reference(x, y, groups, development, test),
        "selected_epochs": epochs,
        "final_training_accuracy": round(float(model.evaluate(x[development], y[development], verbose=0)[1]), 4),
        "test_accuracy": round(float(accuracy_score(y[test], prediction)), 4),
        "test_macro_f1": round(float(f1_score(y[test], prediction, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y[test], prediction).tolist(),
    }
    save_json("cnn_metrics.json", metrics)
    plot_cnn_learning_curves(results["final"]["folds"])
    plot_cnn_folds(results)
    print("CNN test:", metrics["test_accuracy"], metrics["test_macro_f1"], flush=True)
    return metrics


def fit_recurrent(cell, splits):
    train_x, train_y, _ = splits["train"]
    val_x, val_y, _ = splits["validation"]
    keras.backend.clear_session()
    keras.utils.set_random_seed(242)
    model = build_recurrent(cell)
    history = model.fit(
        train_x, train_y, validation_data=(val_x, val_y),
        epochs=55, batch_size=32, shuffle=False, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)],
    )
    return model, history


def train_gru():
    splits, scalers = temporal_splits()
    np.savez(OUTPUT / "gru_scalers.npz", **scalers)
    train_x, train_y, train_times = splits["train"]
    val_x, val_y, val_times = splits["validation"]
    test_x, test_y, test_times = splits["test"]
    assert train_times.max() < val_times.min() < test_times.min()
    mean, std = float(scalers["target_mean"]), float(scalers["target_std"])
    actual = test_y * std + mean

    def forecast(model, inputs):
        return model.predict(inputs, verbose=0).ravel() * std + mean

    comparison = {}
    for cell in RECURRENT_CELLS:
        model, history = fit_recurrent(cell, splits)
        prediction = forecast(model, test_x)
        comparison[cell] = {
            "parameters": model.count_params(),
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "validation_mae": round(float(mean_absolute_error(val_y * std + mean, forecast(model, val_x))), 4),
            "test_mae": round(float(mean_absolute_error(actual, prediction)), 4),
            "test_rmse": round(float(np.sqrt(mean_squared_error(actual, prediction))), 4),
        }
        print(f"{cell} test: {comparison[cell]['test_mae']}", flush=True)
        if cell == "gru":
            gru, gru_history, gru_prediction = model, history, prediction

    gru.save(OUTPUT / "gru.keras")
    gru.save_weights(OUTPUT / "gru.weights.h5")
    plot_history(gru_history, "GRU")
    persistence = test_x[:, -1, 0] * scalers["feature_std"][0] + scalers["feature_mean"][0]
    _, _, expected = temporal_series()
    residuals = np.abs(actual - gru_prediction)
    metrics = {
        "dataset": "deterministic synthetic multivariate autoregressive time series",
        "ranges": {"train": [0, 960], "validation": [960, 1280], "test": [1280, 1600]},
        "window_counts": {name: len(part[0]) for name, part in splits.items()},
        "lookback": train_x.shape[1],
        "best_epoch": comparison["gru"]["best_epoch"],
        "training_mae": round(float(mean_absolute_error(train_y * std + mean, forecast(gru, train_x))), 4),
        "validation_mae": comparison["gru"]["validation_mae"],
        "test_mae": comparison["gru"]["test_mae"],
        "test_rmse": comparison["gru"]["test_rmse"],
        "persistence_test_mae": round(float(mean_absolute_error(actual, persistence)), 4),
        "persistence_test_rmse": round(float(np.sqrt(mean_squared_error(actual, persistence))), 4),
        "noise_floor_test_mae": round(float(mean_absolute_error(actual, expected[test_times])), 4),
        "noise_floor_test_rmse": round(float(np.sqrt(mean_squared_error(actual, expected[test_times]))), 4),
        "test_mae_by_chronological_quarter": [round(float(chunk.mean()), 4) for chunk in np.array_split(residuals, 4)],
        "cell_comparison": comparison,
    }
    save_json("gru_metrics.json", metrics)
    print("GRU test:", metrics["test_mae"], metrics["test_rmse"], flush=True)
    return metrics


if __name__ == "__main__":
    setup()
    train_cnn()
    train_gru()
