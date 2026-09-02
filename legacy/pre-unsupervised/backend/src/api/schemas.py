"""
Pydantic request/response models for the fraud detection API.

Only the ``/api/predict`` endpoint needs a request schema — every other route
returns a precomputed artifact whose shape is documented on the artifact
writer itself (``scripts/train_pipeline.py``) rather than duplicated here.
"""

from pydantic import BaseModel, Field

from backend.src.modeling.config import SELECTED_FEATURES


class TransactionInput(BaseModel):
    """
    A single raw transaction to score for fraud.

    Field names and the accepted values mirror the raw dataset columns
    consumed by ``TransactionDataLoader`` and ``DataPreparation`` — the
    request is transformed through the exact same feature pipeline used in
    training, so no field here is optional or defaulted silently.
    """

    State: str = Field(..., description="Indian state where the transaction occurred")
    Account_Balance: float = Field(..., ge=0, description="Account balance before the transaction")
    Transaction_Amount: float = Field(..., ge=0, description="Transaction amount")
    Merchant_Category: str = Field(..., description="Merchant category, e.g. 'Groceries'")
    Age: int = Field(..., ge=0, le=120, description="Customer age")
    Account_Type: str = Field(..., description="'Savings', 'Checking' or 'Business'")
    Device_Type: str = Field(..., description="'Mobile', 'Desktop', 'POS' or 'ATM'")
    Transaction_Device: str = Field(..., description="Specific device used, e.g. 'ATM Booth Kiosk'")
    Transaction_Date: str = Field(..., description="dd-mm-YYYY")
    Transaction_Time: str = Field(..., description="HH:MM:SS")

    model_config = {
        "json_schema_extra": {
            "example": {
                "State": "Maharashtra",
                "Account_Balance": 52000.0,
                "Transaction_Amount": 4200.0,
                "Merchant_Category": "Electronics",
                "Age": 34,
                "Account_Type": "Savings",
                "Device_Type": "Mobile",
                "Transaction_Device": "QR Code Scanner",
                "Transaction_Date": "15-01-2025",
                "Transaction_Time": "22:41:03",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Fraud score for one scored transaction."""

    fraud_probability: float = Field(..., description="Model's predicted P(Is_Fraud=1)")
    is_fraud_prediction: bool = Field(..., description="fraud_probability >= operating threshold")
    threshold_used: float = Field(..., description="Operating threshold from validation")
    model_name: str = Field(..., description="Estimator that produced the score")
    caveat: str = Field(
        ...,
        description="Restates the model's held-out performance so a score is never read without it",
    )


__all__ = [
    "TransactionInput",
    "PredictionResponse",
    "SELECTED_FEATURES",
]
