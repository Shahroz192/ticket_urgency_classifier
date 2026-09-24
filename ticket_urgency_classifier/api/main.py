from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
import numpy as np
import pandas as pd

from ticket_urgency_classifier.api.models import PredictionResponse, TicketData
from ticket_urgency_classifier.api.utils import (
    label_encoder,
    load_resources,
    model,
    threshold,
    top_tags,
)
from ticket_urgency_classifier.features import (
    add_tag_features,
    engineer_features,
    generate_sentence_transformer_embeddings,
)
from ticket_urgency_classifier.modeling.thresholds import apply_thresholds

# Initialize FastAPI app
app = FastAPI(
    title="Ticket Urgency Classifier API", description="API for predicting ticket urgency levels"
)

# Load resources on startup
load_resources()


@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")


@app.post("/predict", response_model=PredictionResponse)
async def predict(ticket: TicketData):
    """Endpoint to predict ticket urgency using threshold-based classification."""
    # Check if resources are loaded
    if model is None or label_encoder is None or top_tags is None:
        logger.error("Required resources not loaded")
        raise HTTPException(status_code=500, detail="Model resources not available")

    try:
        tags_padded = ticket.tags + [""] * (8 - len(ticket.tags))
        tag_dict = {f"tag_{i + 1}": tag for i, tag in enumerate(tags_padded)}

        df = pd.DataFrame(
            [
                {
                    "subject": ticket.subject,
                    "body": ticket.body,
                    "queue": ticket.queue,
                    "type": ticket.type,
                    "language": ticket.language,
                    **tag_dict,
                }
            ]
        )

        # Apply feature engineering
        logger.info("Applying feature engineering...")
        df = engineer_features(df)
        df = add_tag_features(df, top_tags)

        # Generate sentence transformer embeddings
        logger.info("Generating sentence transformer embeddings...")
        df_embeddings = generate_sentence_transformer_embeddings(df, "")

        df_final = pd.concat([df, df_embeddings], axis=1)

        cols_to_drop = ["subject", "body", "full_text"] + [f"tag_{i}" for i in range(1, 9)]
        df_final.drop(columns=cols_to_drop, errors="ignore", inplace=True)

        logger.info("Performing prediction with threshold...")

        # Get multi-class probabilities (shape: 1 x n_classes)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(df_final)
        else:
            logger.warning("Model doesn't have predict_proba, using predict instead.")
            y_pred_fallback = model.predict(df_final)
            n_classes = len(label_encoder.classes_)
            y_proba = np.zeros((1, n_classes))
            y_proba[0, y_pred_fallback[0]] = 1.0

        class_names = list(label_encoder.classes_)
        pred_idx = int(apply_thresholds(y_proba, threshold, class_names)[0])

        # Confidence score: probability of the predicted class
        confidence_score = float(y_proba[0, pred_idx])
        human_readable_label = label_encoder.inverse_transform([pred_idx])[0]

        logger.success(
            f"Prediction successful: {human_readable_label} with confidence {confidence_score:.2f}"
        )

        return PredictionResponse(
            raw_prediction=int(pred_idx),
            human_readable_label=human_readable_label,
            confidence_score=float(confidence_score),
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
