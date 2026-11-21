# Phishing URL Detector

A machine learning-based phishing URL detection system built with **Python, TensorFlow, and Flask**. It analyzes URLs using 49 engineered lexical features, predicts phishing probability through a neural network, and provides contextual explanations using Retrieval-Augmented Generation (RAG).

The application also includes SQLite-backed prediction history, a REST API, and Docker containerization.

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 89.94% |
| Precision | 89.66% |
| Recall | 90.29% |
| F1-Score | 89.97% |
| ROC-AUC | 96.48% |

*Results from a stratified held-out test set (20%, random seed 42).*

---

## Key Features

- **Neural Network Classification:** TensorFlow/Keras MLP trained on 49 engineered URL features.
- **Feature Engineering:** Extracts lexical and structural URL characteristics.
- **RAG-Based Explanations:** Retrieves relevant phishing guidance using LangChain and FAISS.
- **LLM Integration:** Optional Anthropic integration for generating contextual explanations.
- **Offline Support:** TF-IDF-based retrieval and template explanations work without an external LLM API.
- **Prediction History:** Stores URL checks and classification results in SQLite.
- **REST API:** Flask endpoints for predictions, explanations, history, and statistics.
- **Containerization:** Docker-based application packaging using Gunicorn.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | TensorFlow, Keras, Scikit-learn |
| Data Processing | NumPy, Pandas |
| Backend | Flask, Gunicorn |
| RAG & LLM | LangChain, FAISS, Anthropic API |
| Embeddings | TF-IDF |
| Database | SQLite |
| Deployment | Docker |
| Testing | Pytest, GitHub Actions |

## System Architecture

```text
                 USER INPUT
                     |
                     v
                URL ANALYSIS
                     |
                     v
             FEATURE EXTRACTION
                (49 Features)
                     |
                     v
               StandardScaler
                     |
                     v
              TensorFlow MLP
                     |
                     v
             PHISHING PROBABILITY
                     |
          +----------+-----------+
          |                      |
          v                      v
    SQLite Storage        Warning Signals
          |                      |
          v                      v
    Prediction History    FAISS Retrieval
                                 |
                                 v
                         Relevant Knowledge
                                 |
                                 v
                          Prompt Template
                                 |
                                 v
                         LLM / Offline Mode
                                 |
                                 v
                         EXPLANATION
```

## Machine Learning Model

The classification model is a Multi-Layer Perceptron (MLP) implemented using TensorFlow/Keras.

**Architecture:**

```text
Input Layer:       49 features
        |
Dense Layer:      64 neurons (ReLU)
        |
Dropout:          0.3
        |
Dense Layer:      32 neurons (ReLU)
        |
Dropout:          0.3
        |
Output Layer:     1 neuron (Sigmoid)
```

**Training approach:**
- Stratified train/test split.
- Feature normalization using `StandardScaler`.
- Binary cross-entropy loss.
- Adam optimizer.
- Dropout regularization.
- Early stopping and learning-rate reduction.
- Decision threshold tuning using Youden's J statistic.

### Model Comparison

The following models were evaluated using the same feature set and data split.

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 84.4% | 84.0% | 93.4% |
| Random Forest | 90.0% | 90.0% | 96.2% |
| TensorFlow MLP | 89.9% | 90.0% | 96.5% |

The MLP performs competitively with Random Forest, achieving a slightly higher ROC-AUC while maintaining comparable classification performance.

## RAG-Based Explanation System

The application includes a retrieval pipeline that provides contextual information about phishing indicators.

**How it works:**

1. Extracts warning signals from the analyzed URL.
2. Retrieves relevant passages from the phishing knowledge base.
3. Uses FAISS for similarity-based retrieval.
4. Builds a grounded prompt using LangChain.
5. Generates an explanation using Anthropic when configured.
6. Falls back to offline explanations when an API key is unavailable.

**Implementation details:**
- LangChain `ChatPromptTemplate`
- LangChain `StrOutputParser`
- FAISS vector store
- Custom `TfidfEmbeddings` implementation
- Optional Anthropic LLM integration

The default embedding implementation uses TF-IDF rather than a neural embedding model. This allows retrieval to work without downloading external embedding models.

Classification is performed locally; the optional LLM is used only for explanation generation.

## Project Structure

```text
phishing-url-detector/
│
├── phishing_detector/
│   ├── features.py
│   ├── detector.py
│   ├── storage.py
│   └── rag.py
│
├── web/
│   ├── app.py
│   └── templates/
│
├── knowledge/
│   └── phishing_tactics.md
│
├── data/
│
├── docs/
│   └── deploy-aws.md
│
├── train.py
├── predict.py
├── baselines.py
├── my_model.keras
├── scaler.pkl
├── metrics.json
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python
- pip
- Git (optional)
- Docker (for containerized execution)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/phishing-url-detector.git

cd phishing-url-detector
```

### 2. Create a virtual environment

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m web.app --port 5000
```

Open:

```text
http://localhost:5000
```

### 5. Run a URL prediction from the terminal

```bash
python predict.py "http://secure-account-verify-login.tk/login.html"
```

### 6. Run the model comparison

```bash
python baselines.py
```

### 7. Run tests

```bash
pytest -q
```

**Note:** Running `python train.py` retrains the model and overwrites the existing model artifacts.

## API Reference

The Flask application exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Classifies a submitted URL |
| POST | `/api/explain` | Returns classification and RAG explanation |
| GET | `/api/history` | Retrieves recent prediction history |
| GET | `/api/stats` | Returns prediction statistics |
| GET | `/api/health` | Application health check |

### Example: URL Prediction

**Request**

```http
POST /api/predict
Content-Type: application/json
```

```json
{
  "url": "http://example.com/login"
}
```

### Example: Retrieve Prediction History

```http
GET /api/history?limit=10
```

## Docker

The application can be built and executed inside a Docker container.

### Build the image

```bash
docker build -t phishing-url-detector .
```

### Run the container

```bash
docker run --rm -p 8000:8000 phishing-url-detector
```

Open:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/api/health
```

## Optional: Anthropic API Integration

The RAG explanation pipeline supports optional Anthropic API integration.

Set the following environment variable:

```bash
ANTHROPIC_API_KEY=your_api_key
```

Without an API key, the application continues to provide explanations using its offline retrieval and fallback pipeline.

When enabled, the URL and associated warning signals are sent to the configured LLM provider for explanation generation.

## Limitations

- The classifier relies primarily on lexical and structural URL features.
- It does not inspect website HTML, page content, DNS records, or SSL certificates.
- Compromised legitimate domains and sophisticated brand impersonation can result in incorrect classifications.
- Some URL heuristics may produce false positives on legitimate websites.
- Model performance on the benchmark dataset does not guarantee equivalent performance on previously unseen real-world phishing campaigns.
- Predictions are advisory and should not replace security review.

## Future Improvements

- Improve detection of brand impersonation and deceptive domains.
- Incorporate DNS, SSL, and website-content features.
- Evaluate neural embedding models for semantic retrieval.
- Expand evaluation using newer phishing datasets.
- Improve model generalization and reduce false negatives.
- Deploy the application to a public cloud environment.

## License & Attribution

This project is licensed under the MIT License. See [LICENSE](LICENSE).

The project builds upon an existing open-source phishing detection dataset and feature-engineering foundation, with additional components developed independently, including the TensorFlow classification pipeline, Flask application, SQLite prediction history, RAG-based explanation system, model comparisons, and Docker configuration.