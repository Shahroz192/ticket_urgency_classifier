"""Data validation script for the ticket urgency classifier dataset."""

from loguru import logger
import pandas as pd
import pandera.pandas as pa
import typer

from ticket_urgency_classifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def get_raw_schema() -> pa.DataFrameSchema:
    """
    Returns the pandera schema for the raw dataset.
    """
    schema = pa.DataFrameSchema(
        {
            "subject": pa.Column(str, nullable=True),
            "body": pa.Column(str, nullable=True),
            "queue": pa.Column(str),
            "type": pa.Column(str),
            "language": pa.Column(str),
            "priority": pa.Column(str, checks=pa.Check.isin(["low", "medium", "high"])),
            "version": pa.Column(str, nullable=True),
            "answer": pa.Column(str, nullable=True),
            "tag_1": pa.Column(str, nullable=True),
            "tag_2": pa.Column(str, nullable=True),
            "tag_3": pa.Column(str, nullable=True),
            "tag_4": pa.Column(str, nullable=True),
            "tag_5": pa.Column(str, nullable=True),
            "tag_6": pa.Column(str, nullable=True),
            "tag_7": pa.Column(str, nullable=True),
            "tag_8": pa.Column(str, nullable=True),
        },
        strict=True,
        coerce=True,
    )
    return schema


def get_processed_schema() -> pa.DataFrameSchema:
    """
    Returns the pandera schema for the processed (feature-engineered) dataset.
    This schema is dynamic to handle varying numbers of embedding and tag features.
    """
    schema = pa.DataFrameSchema(
        {
            "queue": pa.Column(str),
            "type": pa.Column(str),
            "language": pa.Column(str),
            "priority": pa.Column(int, checks=pa.Check.isin([0, 1, 2])),
            "sentiment_score": pa.Column(float),
            "word_count": pa.Column(int),
            "exclamation_count": pa.Column(int),
            "question_mark_count": pa.Column(int),
            "urgency_keyword_count": pa.Column(int),
            "question_keyword_count": pa.Column(int),
            "bug_keyword_count": pa.Column(int),
            "queue_type_interaction": pa.Column(str),
            "tag_.*": pa.Column(int, regex=True),
            "emb_.*": pa.Column(float, regex=True),
        },
        strict=False,
        coerce=True,
    )
    return schema


@app.command()
def raw():
    """Validate the raw dataset using pandera."""
    logger.info("Starting raw data validation...")

    dataset_file = RAW_DATA_DIR / "dataset.csv"
    if not dataset_file.exists():
        logger.error(f"Dataset not found at {dataset_file}. Run the dataset download first.")
        raise typer.Exit(code=1)

    df = pd.read_csv(dataset_file)
    schema = get_raw_schema()

    try:
        schema.validate(df, lazy=True)
        logger.success("Raw data validation successful!")
    except pa.errors.SchemaErrors as err:
        logger.error("Raw data validation failed.")
        logger.error(err.failure_cases)
        raise typer.Exit(code=1)


@app.command()
def processed():
    """Validate the processed datasets (train_features.csv and test_features.csv)."""
    logger.info("Starting processed data validation...")

    train_file = PROCESSED_DATA_DIR / "train_features.csv"
    test_file = PROCESSED_DATA_DIR / "test_features.csv"

    if not train_file.exists() or not test_file.exists():
        logger.error("Processed dataset not found. Run the feature engineering first.")
        raise typer.Exit(code=1)

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    schema = get_processed_schema()

    try:
        logger.info("Validating training data...")
        schema.validate(df_train, lazy=True)
        logger.success("Training data validation successful!")

        logger.info("Validating test data...")
        schema.validate(df_test, lazy=True)
        logger.success("Test data validation successful!")

    except pa.errors.SchemaErrors as err:
        logger.error("Processed data validation failed.")
        logger.error(err.failure_cases)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
