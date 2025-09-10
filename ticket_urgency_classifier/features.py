import re

import joblib
from loguru import logger
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import torch
import typer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ticket_urgency_classifier.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SENTENCE_TRANSFORMER_MODEL,
)

app = typer.Typer()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the dataframe.
    - Creates 'full_text' by combining subject and body.
    - Extracts text statistics (sentiment, word count, etc.).
    - Counts occurrences of predefined keywords.
    - Creates an interaction feature between 'queue' and 'type'.
    """
    logger.info("Applying base feature engineering...")

    df["full_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["full_text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )
    df["word_count"] = df["full_text"].apply(lambda x: len(x.split()))
    df["exclamation_count"] = df["full_text"].str.count("!")
    df["question_mark_count"] = df["full_text"].str.count(r"\?")

    # Keyword extraction
    urgency_keywords = [
        "payment",
        "failed",
        "cannot access",
        "login error",
        "outage",
        "urgent",
        "asap",
        "critical",
    ]
    question_keywords = ["how to", "where is", "can you", "inquiry", "question"]
    bug_keywords = ["error code", "exception", "not working", "crash", "bug report", "defect"]

    urgency_regex = r"\b(" + "|".join(urgency_keywords) + r")\b"
    question_regex = r"\b(" + "|".join(question_keywords) + r")\b"
    bug_regex = r"\b(" + "|".join(bug_keywords) + r")\b"

    df["urgency_keyword_count"] = df["full_text"].str.count(urgency_regex, flags=re.IGNORECASE)
    df["question_keyword_count"] = df["full_text"].str.count(question_regex, flags=re.IGNORECASE)
    df["bug_keyword_count"] = df["full_text"].str.count(bug_regex, flags=re.IGNORECASE)

    # Interaction feature
    df["queue_type_interaction"] = df["queue"].astype(str) + "_" + df["type"].astype(str)

    return df


def add_tag_features(df: pd.DataFrame, top_tags_list: list) -> pd.DataFrame:
    """
    Adds binary features for the presence of top tags.
    """
    logger.info("Implementing Tag Intelligence...")
    tag_cols = [f"tag_{i}" for i in range(1, 9)]
    df["all_tags_set"] = df[tag_cols].apply(lambda x: set(x.dropna()), axis=1)

    for tag in top_tags_list:
        col_name = f"tag_{tag.replace(' ', '_')}"
        df[col_name] = df["all_tags_set"].apply(lambda x: 1 if tag in x else 0)

    df.drop(columns="all_tags_set", inplace=True)
    return df


def generate_sentence_transformer_embeddings(
    df: pd.DataFrame, data_split: str, save_to_disk: bool = None
) -> pd.DataFrame:
    """
    Generates Sentence Transformer embeddings for the 'full_text' column.
    Optionally saves embeddings to interim data directory.

    Args:
        df: DataFrame with 'full_text' column
        data_split: Identifier for the dataset ('train', 'test', etc.)
        save_to_disk: Whether to save/load from disk. If None, auto-detects:
                     - Save for 'train'/'test' splits
                     - Don't save for empty string or prediction scenarios
    """

    # Auto-detect behavior if not explicitly specified
    if save_to_disk is None:
        save_to_disk = data_split in ["train", "test"]

    logger.info(f"Generating Sentence Transformer embeddings for {data_split} set...")

    if save_to_disk:
        st_file = INTERIM_DATA_DIR / f"sentence_embeddings_{data_split}.parquet"

        if st_file.exists():
            logger.info(f"Loading cached embeddings from {st_file}...")
            return pd.read_parquet(st_file)

    # Generate embeddings (always in memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=device)

    text_embeddings = embedder.encode(
        df["full_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=save_to_disk,  # Only show progress bar for training
        batch_size=64,
    )

    feature_cols = [f"emb_{i}" for i in range(text_embeddings.shape[1])]
    df_embeddings = pd.DataFrame(text_embeddings, columns=feature_cols)

    if save_to_disk:
        df_embeddings.to_parquet(st_file, index=False)
        logger.info(f"Embeddings saved to {st_file}.")

    return df_embeddings


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all column names are strings, required for ColumnTransformer."""
    df.columns = [str(col) for col in df.columns]
    return df


@app.command()
def main():
    """Main function to orchestrate feature engineering and vectorization."""
    logger.info("Starting comprehensive feature engineering (including vectorization)...")

    # 1. Load data
    train_file = PROCESSED_DATA_DIR / "train.csv"
    test_file = PROCESSED_DATA_DIR / "test.csv"

    if not train_file.exists() or not test_file.exists():
        logger.error("Interim train or test data not found. Run dataset processing first.")
        return

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    logger.info(f"Loaded interim data. Train: {df_train.shape}, Test: {df_test.shape}")

    # 2. Apply feature engineering
    df_train = engineer_features(df_train.copy())
    df_test = engineer_features(df_test.copy())

    # 3. Apply tag intelligence (using top tags from training set)
    tag_cols = [f"tag_{i}" for i in range(1, 9)]
    all_tags_series = df_train[tag_cols].stack()
    top_30_tags = all_tags_series.value_counts().nlargest(30).index.tolist()
    logger.info(f"Top 30 most common tags identified: {top_30_tags}")

    # Save top_30_tags to a file for later use in prediction
    top_tags_file = MODELS_DIR / "top_tags.joblib"
    joblib.dump(top_30_tags, top_tags_file)
    logger.info(f"Top 30 tags saved to {top_tags_file}")

    df_train = add_tag_features(df_train, top_30_tags)
    df_test = add_tag_features(df_test, top_30_tags)

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # 4. Integrated Vectorization: Sentence Transformers
    df_embeddings_train = generate_sentence_transformer_embeddings(df_train, "train", True)
    df_embeddings_test = generate_sentence_transformer_embeddings(df_test, "test", True)

    # 5. Concatenate features
    df_train_final = pd.concat([df_train, df_embeddings_train], axis=1)
    df_test_final = pd.concat([df_test, df_embeddings_test], axis=1)

    cols_to_drop = ["subject", "body", "full_text"] + [f"tag_{i}" for i in range(1, 9)]

    df_train_final.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    df_test_final.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    logger.info("Dropped non-feature columns.")

    target_col = "priority"
    le = LabelEncoder()
    df_train_final[target_col] = le.fit_transform(df_train_final[target_col])
    df_test_final[target_col] = le.transform(df_test_final[target_col])
    logger.info(f"Target classes: {le.classes_}")

    # Save the fitted encoder
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    logger.info("Label encoder saved.")

    # 6. Save fully engineered interim data
    final_train_file = PROCESSED_DATA_DIR / "train_features.csv"
    final_test_file = PROCESSED_DATA_DIR / "test_features.csv"

    df_train_final.to_csv(final_train_file, index=False)
    df_test_final.to_csv(final_test_file, index=False)
    logger.success(
        f"Fully engineered interim data saved. Train: {final_train_file}, Test: {final_test_file}"
    )


if __name__ == "__main__":
    main()
