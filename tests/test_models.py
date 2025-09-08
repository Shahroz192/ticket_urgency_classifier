"""Tests for the entire modeling pipeline."""

from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ticket_urgency_classifier.modeling.predict import (
    app as predict_app,
)
from ticket_urgency_classifier.modeling.train import main as train_main


@pytest.fixture
def mock_model_data_files(tmp_path):
    """Create temporary directories and dummy data files for modeling tests."""
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    df_train_engineered = pd.DataFrame(
        {
            "subject": ["s1"],
            "body": ["b1"],
            "full_text": ["s1 b1"],
            "tag_1": ["t1"],
            "priority": ["low"],
        }
    )
    df_test_engineered = pd.DataFrame(
        {
            "subject": ["s2"],
            "body": ["b2"],
            "full_text": ["s2 b2"],
            "tag_1": ["t2"],
            "priority": ["high"],
        }
    )

    train_engineered_file = interim_dir / "train_engineered_with_embeddings.csv"
    test_engineered_file = interim_dir / "test_engineered_with_embeddings.csv"
    df_train_engineered.to_csv(train_engineered_file, index=False)
    df_test_engineered.to_csv(test_engineered_file, index=False)

    return interim_dir, processed_dir, models_dir


@patch("ticket_urgency_classifier.modeling.train.RandomizedSearchCV")
def test_train_main(mock_random_search, mock_model_data_files):
    """Test the model training script."""
    _, processed_dir, models_dir = mock_model_data_files

    # Create dummy final data for training with enough samples for 3 splits
    df_train_final = pd.DataFrame(
        {
            "priority": [0, 1, 0, 1, 0, 1],
            "language": ["en", "en", "fr", "fr", "en", "fr"],
            "queue": ["q1", "q2", "q1", "q2", "q1", "q2"],
            "type": ["t1", "t2", "t1", "t2", "t1", "t2"],
            "queue_type_interaction": ["q1_t1", "q2_t2", "q1_t1", "q2_t2", "q1_t1", "q2_t2"],
            "numerical_feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    train_final_file = processed_dir / "train_features.csv"
    df_train_final.to_csv(train_final_file, index=False)

    # Create test_features.csv as well
    test_final_file = processed_dir / "test_features.csv"
    df_test_final = pd.DataFrame(
        {
            "priority": [0, 1, 0, 1, 0, 1],
            "language": ["en", "en", "fr", "fr", "en", "fr"],
            "queue": ["q1", "q2", "q1", "q2", "q1", "q2"],
            "type": ["t1", "t2", "t1", "t2", "t1", "t2"],
            "queue_type_interaction": ["q1_t1", "q2_t2", "q1_t1", "q2_t2", "q1_t1", "q2_t2"],
            "numerical_feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    df_test_final.to_csv(test_final_file, index=False)

    # Create label encoder file
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.classes_ = np.array(["low", "medium", "high"])
    joblib.dump(le, models_dir / "label_encoder.joblib")

    # Mock the RandomizedSearchCV to avoid actual training
    # Use PickleableMock to avoid PicklingError
    mock_model = PickleableMock()
    mock_search_instance = MagicMock()
    mock_search_instance.best_estimator_ = mock_model
    mock_search_instance.best_params_ = {"param": "value"}
    mock_search_instance.best_score_ = 0.99
    # Patch fit to avoid actual training
    mock_search_instance.fit = MagicMock()
    mock_random_search.return_value = mock_search_instance

    with (
        patch("ticket_urgency_classifier.modeling.train.PROCESSED_DATA_DIR", processed_dir),
        patch("ticket_urgency_classifier.modeling.train.MODELS_DIR", models_dir),
    ):
        train_main()

        mock_random_search.assert_called_once()
        mock_search_instance.fit.assert_called_once()

        model_file = models_dir / "best_rf_model.joblib"
        assert model_file.exists()
        saved_model = joblib.load(model_file)
        assert saved_model == mock_search_instance.best_estimator_


# A simple, pickleable class to stand in for a real model
class PickleableMock:
    def predict(self, X):
        return np.array([0] * len(X))

    def predict_proba(self, X):
        return np.array([[0.9, 0.1]] * len(X))

    def __eq__(self, other):
        # Make all PickleableMock instances equal to each other for test assertions
        return isinstance(other, PickleableMock)


# A simple, pickleable class to stand in for a real encoder
class PickleableEncoderMock:
    classes_ = np.array(["low", "medium", "high"])

    def transform(self, X):
        return [0] * len(X)

    def inverse_transform(self, X):
        # Map 0 to "low", 1 to "medium", 2 to "high"
        label_map = {0: "low", 1: "medium", 2: "high"}
        return [label_map.get(x, "low") for x in X]


# A simple, pickleable class to stand in for a real model with predict_proba
class PickleableMockWithProba(PickleableMock):
    def predict_proba(self, X):
        import numpy as np

        return np.array([[0.95, 0.05]] * len(X))


def test_evaluate_main(mock_model_data_files):
    """Test the model evaluation script."""
    from ticket_urgency_classifier.modeling.evaluate import (
        main as evaluate_main,
    )

    _, processed_dir, models_dir = mock_model_data_files

    # Create dummy files needed for evaluation
    df_test_final = pd.DataFrame({"priority": [0, 1], "feature1": [0.5, 0.6]})
    test_final_file = processed_dir / "test_features.csv"
    df_test_final.to_csv(test_final_file, index=False)

    mock_model = PickleableMock()
    joblib.dump(mock_model, models_dir / "best_rf_model.joblib")

    mock_encoder = PickleableEncoderMock()
    joblib.dump(mock_encoder, models_dir / "label_encoder.joblib")

    # Create threshold file
    joblib.dump(0.3, models_dir / "best_threshold.joblib")

    with (
        patch("ticket_urgency_classifier.modeling.evaluate.PROCESSED_DATA_DIR", processed_dir),
        patch("ticket_urgency_classifier.modeling.evaluate.MODELS_DIR", models_dir),
    ):
        evaluate_main()

        threshold_file = models_dir / "best_threshold.joblib"
        assert threshold_file.exists()
        best_threshold = joblib.load(threshold_file)
        assert isinstance(best_threshold, (int, float))


@patch("ticket_urgency_classifier.features.generate_sentence_transformer_embeddings")
def test_predict_main(mock_generate_embeddings, tmp_path):
    """Test the prediction script using the Typer CLI runner."""
    # Arrange
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create dummy input file with all expected tag columns
    df_input = pd.DataFrame(
        {
            "subject": ["s1"],
            "body": ["b1"],
            "queue": ["q1"],
            "type": ["t1"],
            "language": ["en"],
            **{f"tag_{i}": [f"t{i}"] for i in range(1, 9)},
        }
    )
    input_file = input_dir / "input.csv"
    df_input.to_csv(input_file, index=False)
    predictions_file = processed_dir / "test_predictions.csv"

    # Create dummy model, tags, and encoder that always decodes 0 to 'low'
    mock_model = PickleableMock()
    joblib.dump(mock_model, models_dir / "best_rf_model.joblib")
    joblib.dump(
        [
            "Feature",
            "Account",
            "Crash",
            "Bug",
            "Support",
            "Customer",
            "Incident",
            "Resolution",
            "Software",
            "Security",
            "Marketing",
            "IT",
            "Recovery",
            "Product",
            "Hardware",
            "Tech_Support",
            "Feedback",
            "Maintenance",
            "Strategy",
            "Guidance",
            "Payment",
            "Disruption",
            "Performance",
            "Outage",
            "Technical",
            "Network",
            "Sales",
            "Documentation",
            "Billing",
            "Integration",
        ],
        models_dir / "top_tags.joblib",
    )
    encoder = PickleableEncoderMock()
    joblib.dump(encoder, models_dir / "label_encoder.joblib")
    joblib.dump(0.5, models_dir / "best_threshold.joblib")

    # Create a mock model with predict_proba method that ensures highest prob for class 0
    mock_model = PickleableMockWithProba()
    joblib.dump(mock_model, models_dir / "best_rf_model.joblib")

    # Mock the embedding generation to avoid downloading a real model
    mock_generate_embeddings.return_value = pd.DataFrame([[0.1, 0.2]])

    with (
        patch("ticket_urgency_classifier.modeling.predict.MODELS_DIR", models_dir),
        patch("ticket_urgency_classifier.modeling.predict.PROCESSED_DATA_DIR", processed_dir),
    ):
        runner = CliRunner()
        # Act
        result = runner.invoke(
            predict_app,
            [
                "batch",
                "--input",
                str(input_file),
                "--top-tags-path",
                str(models_dir / "top_tags.joblib"),
                "--predictions-path",
                str(predictions_file),
            ],
            catch_exceptions=False,
        )

    # Assert
    assert result.exit_code == 0
    assert predictions_file.exists()
    df_preds = pd.read_csv(predictions_file)
    pred_label = df_preds["predicted_urgency"].iloc[0]
    assert pred_label in ["low", "medium", "high"]
