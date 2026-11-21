"""Phishing URL detection package."""

from .detector import PhishingDetector, Prediction
from .features import FEATURE_NAMES, extract_features, extract_features_batch

__all__ = [
    "FEATURE_NAMES",
    "PhishingDetector",
    "Prediction",
    "extract_features",
    "extract_features_batch",
]
