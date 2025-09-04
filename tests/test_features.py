from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ticket_urgency_classifier.features import (
    add_tag_features,
    engineer_features,
    generate_sentence_transformer_embeddings,
)


def test_engineer_features():
    data = {
        "subject": ["Urgent payment failed", "Question about login"],
        "body": ["My payment failed to process.", "I cannot access my account."],
        "queue": ["Payments", "Support"],
        "type": ["Problem", "Question"],
    }
    df = pd.DataFrame(data)
    df_engineered = engineer_features(df)

    assert "full_text" in df_engineered.columns
    assert "sentiment_score" in df_engineered.columns
    assert "word_count" in df_engineered.columns
    assert "exclamation_count" in df_engineered.columns
    assert "question_mark_count" in df_engineered.columns
    assert "urgency_keyword_count" in df_engineered.columns
    assert "question_keyword_count" in df_engineered.columns
    assert "bug_keyword_count" in df_engineered.columns
    assert "queue_type_interaction" in df_engineered.columns

    assert df_engineered["full_text"][0] == "Urgent payment failed My payment failed to process."
    assert df_engineered["word_count"][0] == 8
    assert df_engineered["urgency_keyword_count"][0] == 5
    assert df_engineered["queue_type_interaction"][0] == "Payments_Problem"


def test_add_tag_features():
    data = {
        "tag_1": ["login", "payment"],
        "tag_2": ["error", "failed"],
        "tag_3": [None, "urgent"],
        "tag_4": [None, None],
        "tag_5": [None, None],
        "tag_6": [None, None],
        "tag_7": [None, None],
        "tag_8": [None, None],
    }
    df = pd.DataFrame(data)
    top_tags = ["login", "payment", "urgent"]
    df_tagged = add_tag_features(df, top_tags)

    assert "tag_login" in df_tagged.columns
    assert "tag_payment" in df_tagged.columns
    assert "tag_urgent" in df_tagged.columns

    assert df_tagged["tag_login"][0] == 1
    assert df_tagged["tag_payment"][0] == 0
    assert df_tagged["tag_urgent"][1] == 1


@patch("ticket_urgency_classifier.features.SentenceTransformer")
@patch("ticket_urgency_classifier.features.torch.cuda.is_available")
@patch("ticket_urgency_classifier.features.pd.DataFrame.to_parquet")
@patch("pathlib.Path.exists")
def test_generate_sentence_transformer_embeddings(
    mock_exists, mock_to_parquet, mock_is_available, mock_sentence_transformer
):
    mock_exists.return_value = False
    mock_is_available.return_value = False
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_sentence_transformer.return_value = mock_embedder

    data = {"full_text": ["some text", "some other text"]}
    df = pd.DataFrame(data)
    config = {}
    data_split = "test"

    df_embeddings = generate_sentence_transformer_embeddings(df, config, data_split)

    assert "emb_0" in df_embeddings.columns
    assert "emb_1" in df_embeddings.columns
    assert df_embeddings.shape == (2, 2)
    mock_to_parquet.assert_called_once()
