"""Train the phishing URL detection model.

Usage:
    python train.py [--data dataset_phishing.csv] [--out my_model.keras]

Produces:
    my_model.keras  - trained Keras model
    scaler.pkl      - fitted StandardScaler
    metrics.json    - evaluation metrics + threshold on the held-out test set
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from phishing_detector.data import DATASET_PATH, SEED, load_dataset, split_data

DATA_DEFAULT = DATASET_PATH
OUT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(n_features: int) -> keras.Model:
    inp = keras.Input(shape=(n_features,), name="features")
    x = keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
    )(inp)
    x = keras.layers.Dropout(0.3, seed=SEED)(x)
    x = keras.layers.Dense(
        32, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED + 1),
    )(x)
    x = keras.layers.Dropout(0.3, seed=SEED + 1)(x)
    out = keras.layers.Dense(
        1, activation="sigmoid",
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED + 2),
        name="phishing_prob",
    )(x)
    model = keras.Model(inp, out, name="phishing_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Youden's J statistic on the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def evaluate(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> dict:
    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs >= threshold).astype(np.int32)
    fpr, tpr, roc_thr = roc_curve(y_test, probs)
    precision, recall, pr_thr = precision_recall_curve(y_test, probs)
    report = classification_report(
        y_test, preds, target_names=["legitimate", "phishing"], output_dict=True,
    )
    cm = confusion_matrix(y_test, preds).tolist()
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc_roc": float(roc_auc_score(y_test, probs)),
        "average_precision": float(average_precision_score(y_test, probs)),
        "brier_score": float(brier_score_loss(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    set_seed(SEED)
    print(f"[train] loading dataset from {args.data}")
    X, y = load_dataset(args.data)
    print(f"[train] X shape={X.shape}, y shape={y.shape}, "
          f"positive_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_model(X_train_s.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1,
        ),
    ]

    model.fit(
        X_train_s, y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Threshold tuning on a held-out slice of TRAINING data that the
    # model did not see during fit (the validation_split portion).
    # We approximate this by using the last 1000 training rows, which
    # the validation_split in model.fit() would have withheld.
    n = len(X_train_s)
    val_slice = slice(max(0, n - 1000), n)
    val_probs = model.predict(X_train_s[val_slice], verbose=0).ravel()
    threshold = find_optimal_threshold(y_train[val_slice], val_probs)
    print(f"[train] chosen threshold = {threshold:.4f}")

    metrics = evaluate(model, X_test_s, y_test, threshold)
    print("[train] test metrics:")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("roc_curve", "pr_curve", "classification_report")},
                     indent=2))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "my_model.keras"
    scaler_path = out_dir / "scaler.pkl"
    metrics_path = out_dir / "metrics.json"

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved artifacts to {out_dir}:")
    for p in (model_path, scaler_path, metrics_path):
        print(f"  - {p.name} ({p.stat().st_size} bytes)")



if __name__ == "__main__":
    main()
