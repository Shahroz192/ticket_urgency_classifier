#!/bin/bash
set -e

echo "🚀 Starting app..."

echo "Starting Uvicorn server..."
exec uvicorn ticket_urgency_classifier.api.main:app --host 0.0.0.0 --port 8000
