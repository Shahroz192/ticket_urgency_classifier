from datasets import load_dataset
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

from ticket_urgency_classifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    """Process the raw dataset, clean it, and perform a train/test split."""
    logger.info("Starting data ingestion and processing...")

    # 1. Load the dataset
    dataset_file = RAW_DATA_DIR / "dataset.csv"
    if not dataset_file.exists():
        logger.info("Downloading dataset from Hugging Face...")
        ds = load_dataset("Tobi-Bueck/customer-support-tickets")
        df_raw = ds["train"].to_pandas()
        df_raw.to_csv(dataset_file, index=False)
        logger.info(f"Dataset saved to {dataset_file}.")
    else:
        logger.info(f"Dataset already exists at {dataset_file}. Loading...")
        df_raw = pd.read_csv(dataset_file)

    # 2. Data Cleaning
    logger.info("Cleaning data...")
    df = df_raw.copy()
    df.drop(columns=["version", "answer"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Data shape after cleaning: {df.shape}")

    # 3. Train/Test Split
    logger.info("Performing train/test split...")
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["priority"]
    )
    logger.info(f"Training set size: {df_train.shape}")
    logger.info(f"Test set size: {df_test.shape}")

    # 4. Save interim files
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    train_file = PROCESSED_DATA_DIR / "train.csv"
    test_file = PROCESSED_DATA_DIR / "test.csv"

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)
    logger.success(f"Interim data saved. Train: {train_file}, Test: {test_file}")


if __name__ == "__main__":
    main()
