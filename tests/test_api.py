"""Tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pandas as pd

# Mock dependencies before importing the app to prevent file loading errors
import numpy as np

mock_model = MagicMock()
mock_label_encoder = MagicMock()
mock_label_encoder.classes_ = np.array(['critical', 'high', 'low', 'medium', 'very_low'])
mock_top_tags = ["login", "payment", "urgent"]
mock_threshold = {'critical': 0.26, 'high': 0.36, 'medium': 0.4, 'very_low': 0.3}


# Mock the load_resources function to set the global variables directly
def mock_load_resources():
    pass


# Patch utils' global references *before* importing the FastAPI app
import sys  # noqa: E402

mock_utils = sys.modules.get("ticket_urgency_classifier.api.utils")
if mock_utils is not None:
    mock_utils.model = mock_model
    mock_utils.label_encoder = mock_label_encoder
    mock_utils.top_tags = mock_top_tags
    mock_utils.threshold = mock_threshold
else:
    import ticket_urgency_classifier.api.utils as utils

    utils.model = mock_model
    utils.label_encoder = mock_label_encoder
    utils.top_tags = mock_top_tags
    utils.threshold = mock_threshold

from ticket_urgency_classifier.api.main import app  # noqa: E402

patcher = patch("ticket_urgency_classifier.api.utils.load_resources", mock_load_resources)
patcher.start()


# Stop the patchers after the app is loaded and tests are done
def teardown_module(module):
    """Stop the patchers at the end of the module."""
    patcher.stop()


client = TestClient(app)


@patch("ticket_urgency_classifier.api.main.generate_sentence_transformer_embeddings")
@patch("ticket_urgency_classifier.api.main.add_tag_features")
@patch("ticket_urgency_classifier.api.main.engineer_features")
def test_predict_success(mock_engineer_features, mock_add_tags, mock_generate_embeddings):
    """Test the /predict endpoint for a successful prediction."""
    # Arrange
    # Reset mocks to ensure a clean state for this test
    mock_model.reset_mock()
    mock_label_encoder.reset_mock()

    # Configure mock return values for a successful prediction
    # 5-class probabilities: [critical, high, low, medium, very_low]
    # 'critical' prob (0.85) >= threshold 0.26 → predicts critical
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = np.array([[0.85, 0.05, 0.05, 0.03, 0.02]])
    mock_label_encoder.inverse_transform.return_value = ["critical"]

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
    # critical class is at index 0, its prob 0.85 >= 0.26 threshold
    assert data["raw_prediction"] == 0
    assert data["human_readable_label"] == "critical"
    assert data["confidence_score"] == 0.85
    mock_engineer_features.assert_called_once()
    mock_add_tags.assert_called_once()
    mock_generate_embeddings.assert_called_once()


@patch(
    "ticket_urgency_classifier.api.main.engineer_features",
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
    # It triggers a feature engineering error, not missing resources, so:
    assert response.json() == {"detail": "Internal server error"}
