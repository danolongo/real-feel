# RealFeel

Social media sentiment analysis with integrated bot detection and exclusion.

## Purpose

RealFeel analyzes Twitter sentiment while filtering out bot-generated content. The system combines natural language processing with bot detection to provide authentic sentiment insights from real users.

## What it does

- **Sentiment Analysis**: Classifies tweet emotions using transformer models
- **Bot Detection**: Identifies and filters automated accounts  
- **Data Processing**: Collects, processes, and stores tweet data
- **API Access**: Provides REST endpoints for integration
- **Statistics**: Generates reports and analysis

## Project Structure

```
src/real_feel/
├── textModel/           # Bot detection transformer model
├── pipeline.py          # Main data processing pipeline
├── api.py              # REST API endpoints
├── sentimentPredict.py  # Sentiment analysis
├── twitter_client.py    # Twitter API integration
└── models.py           # Database models
```

## Installation

Requirements:
- Python 3.11+
- Poetry
- PostgreSQL
- Twitter API credentials
- RapidAPI key for Botometer (TEMPORARY)

```bash
git clone <repository-url>
cd real-feel
poetry install
```

## Basic Usage

```python
from real_feel.pipeline import DataPipeline

pipeline = DataPipeline(db_url, twitter_auth, rapidapi_key)
tweets = pipeline.process_tweets("search query", max_tweets=100)
stats = pipeline.get_statistics()
```

## API Server

```bash
poetry run python src/real_feel/db_api.py
```

Available endpoints:
- `POST /analyze` - Process tweets for a query
- `GET /stats` - Get processing statistics  
- `GET /tweets/{sentiment}` - Get tweets by sentiment type

## Documentation Placeholders

The following documentation will be added:

### Core Components
- [Data Pipeline Documentation](docs/pipeline.md)
- [Sentiment Analysis Guide](docs/sentiment.md)
- [Bot Detection Methods](docs/bot-detection.md)
- [API Reference](docs/api.md)

### Model Information
- [Transformer Model Architecture](docs/transformer-model.md)
- [Training and Evaluation](docs/training.md)
- [Performance Benchmarks](docs/metrics.md)

### Deployment
- [Production Setup](docs/deployment.md)
- [Configuration Guide](docs/configuration.md)
- [Monitoring and Logging](docs/monitoring.md)

### Development
- [Contributing Guidelines](docs/contributing.md)
- [Testing Documentation](docs/testing.md)
- [Code Standards](docs/code-standards.md)

## Current Status

This is an active development project. Core functionality is implemented with ongoing improvements to model accuracy and performance.