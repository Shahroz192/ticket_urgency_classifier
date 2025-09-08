#!/bin/bash
set -e

echo "🚀 Starting app..."

if [ "${SKIP_MODEL_DOWNLOAD}" = "true" ]; then
    echo "⚠️  SKIP_MODEL_DOWNLOAD=true: Creating dummy model for testing..."
    mkdir -p /app/models
    touch /app/models/best_rf_model.joblib
else
    echo "📥 Downloading model from S3: ${MODEL_S3_URI} → ${MODEL_PATH}"
    aws s3 cp "${MODEL_S3_URI}" "${MODEL_PATH}"
    if [ ! -s "${MODEL_PATH}" ]; then
        echo "❌ Model file is empty or not downloaded!"
        exit 1
    fi
    echo "✅ Model downloaded and verified."
fi

echo "Starting Uvicorn server..."
exec uvicorn ticket_urgency_classifier.api.main:app --host 0.0.0.0 --port 8000
