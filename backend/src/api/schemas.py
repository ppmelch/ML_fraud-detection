"""
Pydantic request/response models for the anomaly-detection API.

Only ``/api/predict`` needs a request schema — every other route returns a
precomputed artifact whose shape is documented on the artifact writer
(``scripts/train_pipeline.py``) rather than duplicated here.
"""

from pydantic import BaseModel, Field

from backend.src.modeling.config import RAW_FEATURES


class TransactionInput(BaseModel):
    """
    A single raw transaction to score for anomalousness.

    Field names mirror the raw dataset columns consumed by
    ``TransactionDataLoader`` and ``FeatureEngineer``. The request is run
    through the exact feature pipeline used in training, so no field here is
    silently defaulted except the two genuinely optional identifiers.
    """

    TransactionAmount: float = Field(..., ge=0, description="Transaction amount")
    TransactionType: str = Field(..., description="'Debit' or 'Credit'")
    Location: str = Field(..., description="US city name, e.g. 'San Diego'")
    Channel: str = Field(..., description="'ATM', 'Online' or 'Branch'")
    CustomerAge: int = Field(..., ge=0, le=120, description="Customer age")
    CustomerOccupation: str = Field(
        ..., description="'Doctor', 'Student', 'Retired' or 'Engineer'"
    )
    TransactionDuration: float = Field(
        ..., ge=0, description="Transaction duration in seconds"
    )
    LoginAttempts: int = Field(..., ge=0, description="Login attempts before the transaction")
    AccountBalance: float = Field(..., description="Account balance before the transaction")
    TransactionDate: str = Field(..., description="'%Y-%m-%d %H:%M:%S'")

    AccountID: str | None = Field(
        default=None, description="Account identifier; enables per-account features"
    )
    PreviousTransactionDate: str | None = Field(
        default=None,
        description="Accepted but ignored — a data-extraction artefact, never a feature",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmount": 120.0,
                "TransactionType": "Debit",
                "Location": "San Diego",
                "Channel": "ATM",
                "CustomerAge": 70,
                "CustomerOccupation": "Doctor",
                "TransactionDuration": 80.0,
                "LoginAttempts": 1,
                "AccountBalance": 5000.0,
                "TransactionDate": "2023-08-15 14:20:03",
                "AccountID": "AC00128",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Anomaly score for one scored transaction."""

    anomaly_score: float = Field(..., description="Model anomaly score in [0, 1]")
    is_anomaly: bool = Field(..., description="anomaly_score >= operating threshold")
    threshold_used: float = Field(..., description="Contamination-based operating threshold")
    model_name: str = Field(..., description="Estimator that produced the score")
    percentile: float = Field(
        ..., description="Rank of this score against the training score distribution (0-100)"
    )
    caveat: str = Field(
        ...,
        description="Restates that the dataset is unlabelled and the score is not a fraud probability",
    )


__all__ = ["TransactionInput", "PredictionResponse", "RAW_FEATURES"]
