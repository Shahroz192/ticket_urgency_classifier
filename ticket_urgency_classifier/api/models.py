from typing import List

from pydantic import BaseModel, field_validator


class TicketData(BaseModel):
    """Model for input ticket data."""

    subject: str
    body: str
    queue: str
    type: str
    language: str
    tags: List[str]

    @field_validator("subject", "body")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v

    @field_validator("tags")
    def validate_tags(cls, v):
        if len(v) > 8:
            raise ValueError("Tags list cannot exceed 8 items")
        return v


class PredictionResponse(BaseModel):
    """Model for prediction response."""

    raw_prediction: int
    human_readable_label: str
    confidence_score: float
