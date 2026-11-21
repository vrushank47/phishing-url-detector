"""Tests for the SQL store, the RAG explainer (offline mode) and the web API."""

from __future__ import annotations

import pytest

from phishing_detector import Prediction
from phishing_detector.rag import PhishingExplainer, load_knowledge
from phishing_detector.signals import triggered_signals
from phishing_detector.storage import PredictionStore


@pytest.fixture
def store(tmp_path):
    return PredictionStore(tmp_path / "test.db")


def test_store_roundtrip_and_stats(store):
    store.add(Prediction("http://a.tk/login", True, 0.97, 0.5))
    store.add(Prediction("https://b.com", False, 0.03, 0.5))
    store.add(Prediction("http://c.xyz/verify", True, 0.91, 0.5))
    assert [r["url"] for r in store.recent(2)] == ["http://c.xyz/verify", "https://b.com"]
    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_label"]["phishing"]["count"] == 2
    assert len(store.to_dataframe()) == 3


def test_store_is_safe_against_sql_injection(store):
    nasty = "http://x.com/'); DROP TABLE predictions;--"
    store.add(Prediction(nasty, True, 0.9, 0.5))
    assert store.recent(1)[0]["url"] == nasty


def test_knowledge_base_loads():
    assert len(load_knowledge()) >= 10


def test_signals_flag_obvious_phishing():
    s = " ".join(triggered_signals("http://192.168.0.1/secure-login@evil.tk"))
    assert "IP address" in s and "'@'" in s


def test_retrieval_finds_relevant_passage(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ex = PhishingExplainer()
    titles = [d.metadata["title"] for d in ex.retrieve("host is a raw IP address")]
    assert titles[0].startswith("Raw IP")


def test_offline_explanation(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ex = PhishingExplainer()
    out = ex.explain(Prediction("http://secure-login.tk/verify", True, 0.98, 0.5))
    assert ex.mode == "offline" and out["mode"] == "offline"
    assert out["sources"] and "phishing" in out["explanation"]


def test_web_api(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from web import app as webapp

    webapp._store = PredictionStore(tmp_path / "web.db")
    webapp._explainer = None
    client = webapp.app.test_client()

    r = client.post("/api/predict", json={"url": "http://secure-account-verify-login.tk/login.html"})
    assert r.status_code == 200 and r.get_json()["label"] == "phishing"
    assert client.get("/api/history").get_json()[0]["id"] == r.get_json()["id"]
    assert client.get("/api/stats").get_json()["total"] == 1

    r = client.post("/api/explain", json={"url": "https://www.google.com"})
    assert r.status_code == 200 and r.get_json()["explanation"]

    assert client.post("/api/predict", json={"url": ""}).status_code == 400
    assert client.post("/api/predict", json={"url": 5}).status_code == 400
    assert client.get("/").status_code == 200
