"""Load trained artifacts and run phishing predictions on URLs."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow import keras

from .features import extract_features

_REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = _REPO / "my_model.keras"
SCALER_PATH = _REPO / "scaler.pkl"
METRICS_PATH = _REPO / "metrics.json"


@dataclass
class Prediction:
    url: str
    is_phishing: bool
    probability: float
    threshold: float

    @property
    def label(self) -> str:
        return "phishing" if self.is_phishing else "legitimate"

    def as_dict(self) -> dict:
        return {
            "url": self.url,
            "is_phishing": self.is_phishing,
            "label": self.label,
            "probability": round(self.probability, 4),
            "threshold": round(self.threshold, 4),
        }


class PhishingDetector:
    """Keras MLP + StandardScaler + tuned threshold, ready for inference."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        metrics_path: Path = METRICS_PATH,
    ) -> None:
        self.model = keras.models.load_model(model_path, compile=False)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(metrics_path) as f:
            self.metrics = json.load(f)
        self.threshold = float(self.metrics.get("threshold", 0.5))

    def predict_batch(self, urls: list[str]) -> list[Prediction]:
        if not urls:
            return []
        feats = np.vstack([extract_features(u) for u in urls])
        feats_s = self.scaler.transform(feats).astype(np.float32)
        probs = self.model.predict(feats_s, verbose=0).ravel()
        return [
            Prediction(u, float(p) >= self.threshold, float(p), self.threshold)
            for u, p in zip(urls, probs)
        ]

    def predict(self, url: str) -> Prediction:
        return self.predict_batch([url])[0]
