# Electricity Demand Forecasting Backend

A comprehensive ML backend system for predicting electricity consumption using Python, FastAPI, and time-series machine learning models.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Run the API server
uvicorn src.api.main:app --reload --port 8000

# View API docs
open http://localhost:8000/docs
```

## Project Structure

```
├── src/
│   ├── config.py              # Configuration management
│   ├── database/              # Database models and connection
│   ├── data_pipeline/         # Data ingestion and feature engineering
│   ├── ml_pipeline/           # Model training and evaluation
│   ├── api/                   # FastAPI application
│   ├── orchestration/         # Scheduled tasks
│   └── monitoring/            # Drift detection and logging
├── data/
│   ├── raw/                   # Raw input data
│   ├── processed/             # Feature-engineered data
│   └── models/                # Saved model artifacts
├── tests/                     # Unit and integration tests
├── scripts/                   # CLI utilities
└── notebooks/                 # Jupyter notebooks for exploration
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Generate electricity demand forecast |
| `/models` | GET | List available models |
| `/models/{id}` | GET | Get model details |
| `/retrain` | POST | Trigger model retraining |
| `/metrics` | GET | Get performance metrics |

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Docker

```bash
# Build image
docker build -t electricity-forecast .

# Run container
docker run -p 8000:8000 electricity-forecast

# Docker Compose (with PostgreSQL)
docker-compose up -d
```

## Configuration

Set environment variables or create `.env` file:

```env
DATABASE_URL=sqlite:///./data/electricity.db
MODEL_DIR=./data/models
LOG_LEVEL=INFO
```

## License

MIT
# ml-workshop
# ml-workshop
