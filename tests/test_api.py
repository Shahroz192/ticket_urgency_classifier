"""Tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pandas as pd

from ticket_urgency_classifier.api.main import app

# Mock dependencies before importing the app to prevent file loading errors
mock_model = MagicMock()
mock_label_encoder = MagicMock()
mock_top_tags = ["login", "payment", "urgent"]

# Patch joblib.load before the app is imported
patcher = patch("joblib.load", side_effect=[mock_model, mock_label_encoder, mock_top_tags])
patcher.start()


# Stop the patcher after the app is loaded and tests are done
def teardown_module(module):
    """Stop the patcher at the end of the module."""
    patcher.stop()


client = TestClient(app)


@patch("ticket_urgency_classifier.api.generate_sentence_transformer_embeddings")
@patch("ticket_urgency_classifier.api.add_tag_features")
@patch("ticket_urgency_classifier.api.engineer_features")
def test_predict_success(mock_engineer_features, mock_add_tags, mock_generate_embeddings):
    """Test the /predict endpoint for a successful prediction."""
    # Arrange
    # Reset mocks to ensure a clean state for this test
    mock_model.reset_mock()
    mock_label_encoder.reset_mock()

    # Configure mock return values for a successful prediction
    mock_model.predict.return_value = [0]  # Raw prediction for 'low'
    mock_model.predict_proba.return_value.max.return_value = 0.95  # Confidence score
    mock_label_encoder.inverse_transform.return_value = ["low"]  # Human-readable label

    # Mock the feature engineering functions to return dummy dataframes
    mock_engineer_features.return_value = pd.DataFrame([{"full_text": "test"}])
    mock_add_tags.return_value = pd.DataFrame([{"full_text": "test", "tag_login": 1}])
    mock_generate_embeddings.return_value = pd.DataFrame([[0.1, 0.2]])

    ticket_data = {
        "subject": "Cannot login",
        "body": "I am unable to access my account.",
        "queue": "Support",
        "type": "Problem",
        "language": "en",
        "tags": ["login", "error"],
    }

    # Act
    response = client.post("/predict", json=ticket_data)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["raw_prediction"] == 0
    assert data["human_readable_label"] == "low"
    assert data["confidence_score"] == 0.95
    mock_engineer_features.assert_called_once()
    mock_add_tags.assert_called_once()
    mock_generate_embeddings.assert_called_once()


@patch(
    "ticket_urgency_classifier.api.engineer_features",
    side_effect=Exception("Feature Engineering Failed"),
)
def test_predict_internal_error(mock_engineer_features):
    """Test the /predict endpoint for a 500 internal server error."""
    # Arrange
    ticket_data = {
        "subject": "Error case",
        "body": "This should fail.",
        "queue": "Support",
        "type": "Problem",
        "language": "en",
        "tags": ["fail"],
    }

    # Act
    response = client.post("/predict", json=ticket_data)

    # Assert
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
