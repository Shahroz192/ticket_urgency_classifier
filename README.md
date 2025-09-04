# Ticket Urgency Classifier

[![CI](https://github.com/shahroz192/ticket_urgency_classifier/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/shahroz192/ticket_urgency_classifier/actions)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/r/shahroz192/ticket-urgency-classifier)
[![License](https://img.shields.io/github/license/shahroz192/ticket_urgency_classifier)](LICENSE)

Classifies tickets based on the priority in classes: High, Medium, Low

## Description

The Ticket Urgency Classifier is a machine learning system that automatically categorizes customer support tickets into priority levels (High, Medium, Low) based on their content and metadata. This helps support teams efficiently triage tickets and respond to urgent issues faster.

The system uses advanced natural language processing techniques and machine learning algorithms to analyze ticket subjects, bodies, and other metadata to determine urgency. It's designed to be easily deployable and integrable into existing support workflows.

## Key Features

- **Multi-class Classification**: Accurately classifies tickets into High, Medium, and Low priority categories
- **Advanced NLP**: Uses Sentence Transformers for semantic understanding of ticket content
- **Feature Engineering**: Extracts meaningful features from ticket metadata, keywords, and sentiment analysis
- **RESTful API**: Provides a FastAPI-based web service for real-time predictions
- **Batch Processing**: Supports bulk classification of tickets via CSV files
- **MLflow Integration**: Tracks experiments, manages models, and provides model registry
- **Containerization**: Docker support for easy deployment and scalability

## Tech Stack

- **Language**: Python 3.10
- **ML Frameworks**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **NLP**: Sentence Transformers, Hugging Face Transformers
- **Data Processing**: Pandas, Polars, NumPy
- **Visualization**: Seaborn, Matplotlib
- **API**: FastAPI
- **Experiment Tracking**: MLflow
- **Deployment**: Docker
- **Testing**: Pytest
- **CI/CD**: GitHub Actions

## Installation and Setup

### Prerequisites

- Python 3.10
- Docker (optional, for containerized deployment)
- uv (for fast dependency installation)

### Clone the Repository

```bash
git clone https://github.com/shahroz192/ticket_urgency_classifier.git
cd ticket_urgency_classifier
```

### Install Dependencies

Using uv (recommended for faster installation):

```bash
# Create virtual environment
make create_environment

# Activate virtual environment
source ./.venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
make requirements
```

Or using pip:

```bash
pip install -r requirements.txt
```

### Configuration

The project uses a `config.yaml` file for configuration. You can modify paths and hyperparameters in `ticket_urgency_classifier/config.yaml`.

## Usage

### Quick Start with Docker

```bash
# Build the Docker image
docker build -t ticket-urgency-classifier .

# Run the container
docker run -p 8000:8000 ticket-urgency-classifier

# Access the API at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Train a Model

1. Prepare the data:
   ```bash
   make data
   make features
   ```

2. Train the model:
   ```bash
   make model
   ```

### Run Predictions

#### Single Prediction via CLI

```bash
python -m ticket_urgency_classifier modeling predict single \
  --subject "Server down" \
  --body "The main server is not responding to requests." \
  --queue "IT" \
  --type "Problem" \
  --language "en" \
  --tag "server" \
  --tag "critical"
```

#### Batch Predictions via CLI

```bash
python -m ticket_urgency_classifier modeling predict batch \
  --input path/to/your/tickets.csv
```

#### API Predictions

Start the API server:

```bash
make serve-api
```

Then make a POST request to the `/predict` endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot login",
    "body": "I am unable to access my account.",
    "queue": "Support",
    "type": "Problem",
    "language": "en",
    "tags": ["login", "error"]
  }'
```

## Experiment Tracking & Model Registry

This project uses MLflow for experiment tracking and model management:

- **Tracking**: Logs hyperparameters, metrics, and model artifacts during training
- **UI**: View experiments with `mlflow ui` (runs on http://localhost:5000 by default)
- **Model Registry**: Store and version trained models for production use

### Setting up MLflow Server

For persistent storage and model registry:

```bash
# Start MLflow server with SQLite backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Or use a different backend (PostgreSQL, etc.)
mlflow server --backend-store-uri postgresql://user:password@localhost/mlflow_db
```

### Training with Registry

```bash
# Train model (automatically registers to registry)
python -m ticket_urgency_classifier modeling train
```

### Using Registered Models

```bash
# Batch predictions using registry model
python -m ticket_urgency_classifier modeling predict batch --input data.csv

# Serve model via REST API
mlflow models serve -m "models:/ticket_urgency_classifier/Production" -p 5001
```

## Running Tests

To run the test suite:

```bash
make test
```

Or directly with pytest:

```bash
python -m pytest tests/ -v
```

## Contributing

We welcome contributions to the Ticket Urgency Classifier! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure nothing is broken (`make test`)
5. Commit your changes (`git commit -am 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Create a new Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

- **Linting**: Code formatting and style checks with Ruff
- **Testing**: Automated test suite execution
- **Docker Build**: Container image creation and validation
- **Deployment**: Automatic push to Docker Hub on main branch updates

### Setup Requirements

1. **GitHub Secrets**: Add the following secrets to your repository:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub password or access token

2. **Docker Hub**: Ensure you have a Docker Hub account and repository created

The pipeline runs on every push to `main` branch and pull requests, ensuring code quality and automated deployment.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── Dockerfile         <- Docker configuration for containerizing the application
├── .dockerignore      <- Files to exclude from Docker build context
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         ticket_urgency_classifier and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── .github
│   └── workflows
│       └── main.yaml  <- GitHub Actions CI/CD pipeline configuration
│
└── ticket_urgency_classifier   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ticket_urgency_classifier a Python module
    │
    ├── api.py                  <- FastAPI application for serving predictions
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
