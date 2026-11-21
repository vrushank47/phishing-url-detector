"""Compare classic scikit-learn models with the Keras MLP on the same split.

Usage: python baselines.py
Writes baselines.json and prints a comparison table. The MLP numbers are
read from metrics.json (written by train.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phishing_detector.data import SEED, load_dataset, split_data

ROOT = Path(__file__).resolve().parent


def main() -> None:
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = split_data(X, y)

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    rows = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        rows[name] = {
            "accuracy": accuracy_score(y_te, preds),
            "f1": f1_score(y_te, preds),
            "auc_roc": roc_auc_score(y_te, probs),
        }

    mlp = json.loads((ROOT / "metrics.json").read_text())
    rows["Keras MLP (TensorFlow)"] = {k: mlp[k] for k in ("accuracy", "f1", "auc_roc")}

    (ROOT / "baselines.json").write_text(json.dumps(rows, indent=2))
    print(f"{'model':<26}{'accuracy':>10}{'f1':>8}{'auc':>8}")
    for name, r in rows.items():
        print(f"{name:<26}{r['accuracy']:>10.4f}{r['f1']:>8.4f}{r['auc_roc']:>8.4f}")


if __name__ == "__main__":
    main()
