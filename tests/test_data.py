"""Tests for the data processing and splitting script."""

from unittest.mock import patch

import pandas as pd
import pytest

from ticket_urgency_classifier.dataset import main as process_dataset
from ticket_urgency_classifier.validate import get_raw_schema


def test_raw_schema_allows_null_type():
    """Raw source ships ~13k rows with null type; schema must accept them."""
    df = pd.DataFrame(
        {
            "subject": ["s1", "s2"],
            "body": ["b1", "b2"],
            "queue": ["q", "q"],
            "type": [None, "Incident"],
            "language": ["de", "en"],
            "priority": ["critical", "high"],
            "version": [None, None],
            "answer": [None, None],
            "tag_1": [None, None],
            "tag_2": [None, None],
            "tag_3": [None, None],
            "tag_4": [None, None],
            "tag_5": [None, None],
            "tag_6": [None, None],
            "tag_7": [None, None],
            "tag_8": [None, None],
        }
    )
    get_raw_schema().validate(df, lazy=True)


def test_dataset_fills_null_type_with_unknown(mock_dirs):
    """Null types must become 'unknown' — never dropped (would erase critical/very_low)."""
    raw_dir, interim_dir = mock_dirs
    mock_df_raw = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(20)],
            "body": [f"b{i}" for i in range(20)],
            "type": [None if i % 2 == 0 else "Incident" for i in range(20)],
            "priority": ([0] * 7) + ([1] * 7) + ([2] * 6),
            "version": [1.0] * 20,
            "answer": [f"a{i}" for i in range(20)],
        }
    )
    mock_df_raw.to_csv(raw_dir / "dataset.csv", index=False)

    with (
        patch("ticket_urgency_classifier.dataset.load_dataset"),
        patch("ticket_urgency_classifier.dataset.RAW_DATA_DIR", raw_dir),
        patch("ticket_urgency_classifier.dataset.INTERIM_DATA_DIR", interim_dir),
    ):
        process_dataset()

    combined = pd.concat(
        [
            pd.read_csv(interim_dir / "train.csv"),
            pd.read_csv(interim_dir / "test.csv"),
        ],
        ignore_index=True,
    )
    assert len(combined) == 20
    assert combined["type"].isna().sum() == 0
    assert set(combined["type"]) <= {"unknown", "Incident"}


@pytest.fixture
def mock_dirs(tmp_path):
    """Create temporary directories for raw and interim data for testing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    return raw_dir, interim_dir


@patch("ticket_urgency_classifier.dataset.load_dataset")
def test_process_dataset_downloads_and_splits(mock_load_dataset, mock_dirs):
    """Test that the script downloads, processes, and splits data when no local file exists."""
    # Arrange
    raw_dir, interim_dir = mock_dirs

    # Mock the raw dataset that would be downloaded from Hugging Face
    # Ensure enough samples for stratification
    mock_df_raw = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(20)],
            "body": [f"b{i}" for i in range(20)],
            "priority": ([0] * 7) + ([1] * 7) + ([2] * 6),  # Ensure at least 2 of each class
            "version": [1.0] * 20,
            "answer": [f"a{i}" for i in range(20)],
        }
    )
    # Configure the mock to return a real DataFrame
    mock_load_dataset.return_value.__getitem__.return_value.to_pandas.return_value = mock_df_raw

    # Patch the config paths to use our temporary directories
    with (
        patch("ticket_urgency_classifier.dataset.RAW_DATA_DIR", raw_dir),
        patch("ticket_urgency_classifier.dataset.INTERIM_DATA_DIR", interim_dir),
    ):
        # Act
        process_dataset()

        # Assert
        mock_load_dataset.assert_called_once()

        # Check that the raw and interim files were created
        dataset_file = raw_dir / "dataset.csv"
        train_file = interim_dir / "train.csv"
        test_file = interim_dir / "test.csv"

        assert dataset_file.exists()
        assert train_file.exists()
        assert test_file.exists()

        # Verify the contents of the saved files
        df_saved_raw = pd.read_csv(dataset_file)
        pd.testing.assert_frame_equal(df_saved_raw, mock_df_raw)

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        assert len(df_train) == 16  # 80% of 20
        assert len(df_test) == 4  # 20% of 20
        assert "version" not in df_train.columns
        assert "answer" not in df_test.columns


@patch("pandas.read_csv")
@patch(
    "ticket_urgency_classifier.dataset.load_dataset"
)  # Also patch this to ensure it's not called
def test_process_dataset_uses_existing_file(mock_load_dataset, mock_read_csv, mock_dirs):
    """Test that the script uses an existing local file instead of downloading."""
    # Arrange
    raw_dir, interim_dir = mock_dirs
    dataset_file = raw_dir / "dataset.csv"
    dataset_file.touch()  # Create a dummy file to simulate it already existing

    mock_df_raw = pd.DataFrame(
        {
            "subject": [f"s{i}" for i in range(20)],
            "body": [f"b{i}" for i in range(20)],
            "priority": ([0] * 7) + ([1] * 7) + ([2] * 6),
            "version": [1.0] * 20,
            "answer": [f"a{i}" for i in range(20)],
        }
    )
    mock_read_csv.return_value = mock_df_raw

    with (
        patch("ticket_urgency_classifier.dataset.RAW_DATA_DIR", raw_dir),
        patch("ticket_urgency_classifier.dataset.INTERIM_DATA_DIR", interim_dir),
    ):
        # Act
        process_dataset()

        # Assert
        mock_load_dataset.assert_not_called()
        mock_read_csv.assert_called_once_with(dataset_file)

        train_file = interim_dir / "train.csv"
        test_file = interim_dir / "test.csv"

        assert train_file.exists()
        assert test_file.exists()
