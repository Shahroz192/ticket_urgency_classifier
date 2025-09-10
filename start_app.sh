#!/bin/bash
set -e

echo "🚀 Starting app..."

# The model loading logic is now handled within the Python application.
# The app will use the MODEL_S3_URI environment variable to download the model
# using boto3, which will automatically use the EC2 instance's IAM role.

echo "Starting Uvicorn server..."
exec uvicorn ticket_urgency_classifier.api.main:app --host 0.0.0.0 --port 8000
