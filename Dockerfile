FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml .
COPY requirements-prod.txt .
COPY README.md .
COPY LICENSE .
COPY ticket_urgency_classifier/ ./ticket_urgency_classifier/
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -r requirements-prod.txt


FROM python:3.10-slim

WORKDIR /app

# Copy the entire virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY ticket_urgency_classifier/ ./ticket_urgency_classifier/
COPY templates/ ./templates/
COPY models/label_encoder.joblib ./models/
COPY models/top_tags.joblib ./models/
COPY models/best_threshold.joblib ./models/

# Create and switch to a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

ENV MODEL_S3_URI="s3://ticket-classification-ml-models-bucket/ticket-urgency/v1/best_rf_model.joblib"
ENV MODEL_PATH="/app/models/best_rf_model.joblib"
ENV USE_LOCAL_MODEL="false"

EXPOSE 8000

# Use the python module approach to run uvicorn
CMD ["python", "-m", "uvicorn", "ticket_urgency_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
