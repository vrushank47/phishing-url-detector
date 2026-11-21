FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PHISHING_DB=/app/data/predictions.db

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what inference needs: code, knowledge base and trained artifacts.
COPY phishing_detector ./phishing_detector
COPY web ./web
COPY knowledge ./knowledge
COPY my_model.keras scaler.pkl metrics.json ./

RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health')"

# One worker: the TensorFlow model is loaded once per worker process.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8000} web.app:app"]
