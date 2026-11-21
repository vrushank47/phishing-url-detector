"""Flask web UI + JSON API for classifying URLs.

Run locally:   python -m web.app --port 5000
In production: gunicorn web.app:app   (see Dockerfile)

Endpoints:
    POST /api/predict   {"url": ...}  -> verdict (also logged to SQLite)
    POST /api/explain   {"url": ...}  -> verdict + RAG explanation
    GET  /api/history?limit=10        -> recent checks
    GET  /api/stats                   -> counts by label (SQL GROUP BY)
    GET  /api/health
"""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from phishing_detector import PhishingDetector
from phishing_detector.rag import PhishingExplainer
from phishing_detector.storage import PredictionStore

MAX_URL_LENGTH = 2048

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
_detector: PhishingDetector | None = None
_explainer: PhishingExplainer | None = None
_store: PredictionStore | None = None


def get_detector() -> PhishingDetector:
    global _detector
    if _detector is None:
        _detector = PhishingDetector()
    return _detector


def get_explainer() -> PhishingExplainer:
    global _explainer
    if _explainer is None:
        _explainer = PhishingExplainer()
    return _explainer


def get_store() -> PredictionStore:
    global _store
    if _store is None:
        _store = PredictionStore()
    return _store


def _read_url():
    """Validate the request body. Returns (url, None) or (None, error_response)."""
    data = request.get_json(silent=True) or {}
    raw = data.get("url")
    if not isinstance(raw, str):
        return None, (jsonify({"error": "'url' must be a string"}), 400)
    url = raw.strip()
    if not url:
        return None, (jsonify({"error": "Missing 'url' field"}), 400)
    if len(url) > MAX_URL_LENGTH:
        return None, (jsonify({"error": f"URL exceeds maximum length of {MAX_URL_LENGTH} characters"}), 400)
    return url, None


@app.route("/")
def index():
    return render_template(
        "index.html", metrics=get_detector().metrics, explain_mode=get_explainer().mode
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    url, err = _read_url()
    if err:
        return err
    result = get_detector().predict(url)
    out = result.as_dict()
    out["id"] = get_store().add(result)
    return jsonify(out)


@app.route("/api/explain", methods=["POST"])
def api_explain():
    url, err = _read_url()
    if err:
        return err
    result = get_detector().predict(url)
    out = result.as_dict()
    out.update(get_explainer().explain(result))
    return jsonify(out)


@app.route("/api/history")
def api_history():
    limit = min(max(request.args.get("limit", 10, type=int), 1), 100)
    return jsonify(get_store().recent(limit))


@app.route("/api/stats")
def api_stats():
    return jsonify(get_store().stats())


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    get_detector()  # warm up so the first request isn't slow
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
