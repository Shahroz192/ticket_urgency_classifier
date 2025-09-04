# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Copy the application code
COPY ticket_urgency_classifier/ ./ticket_urgency_classifier/

# Copy templates directory
COPY templates/ ./templates/

# Copy models directory
COPY models/label_encoder.joblib ./models/
COPY models/top_tags.joblib ./models/
COPY models/best_threshold.joblib ./models/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
    && chown -R app:app /app
USER app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "ticket_urgency_classifier.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
