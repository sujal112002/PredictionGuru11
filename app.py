import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODELS_DIR = Path("models")
METADATA_PATH = MODELS_DIR / "metadata.json"
MODEL_PATH = MODELS_DIR / "best_model.joblib"


class UserFeaturesInput(BaseModel):
    total_events: float = Field(ge=0)
    total_views: float = Field(ge=0)
    total_addtocart: float = Field(ge=0)
    add_to_view_ratio: float = Field(ge=0)
    activity_level: float = Field(ge=0)
    total_sessions: float = Field(ge=1)
    click_rate: float = Field(ge=0)
    session_duration: float = Field(ge=0)
    pages_per_session: float = Field(ge=0)
    time_since_last_activity: float = Field(ge=0)
    active_days: float = Field(ge=0)
    cart_to_view_rate: float = Field(ge=0)
    avg_events_per_session: float = Field(ge=0)


def _load_inference_artifacts() -> tuple[object, dict]:
    if not METADATA_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python src/train.py` first."
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    return model, metadata


model, metadata = _load_inference_artifacts()
feature_columns = metadata["feature_columns"]
decision_threshold = float(metadata.get("decision_threshold", 0.6))

app = FastAPI(title="User Conversion Prediction API", version="1.0.0")


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model": metadata["best_model_name"],
        "decision_threshold": decision_threshold,
    }


def build_prediction_reason(user_features: UserFeaturesInput, prediction: int) -> str:
    """
    Explanation aligned with model prediction.
    """

    # Positive signals
    high_intent = user_features.add_to_view_ratio >= 0.25
    high_engagement = user_features.session_duration >= 100
    recent_activity = user_features.time_since_last_activity <= 30

    # Negative signals
    low_activity = user_features.activity_level <= 3
    no_cart = user_features.total_addtocart == 0

    if prediction == 1:
        if high_intent:
            return "Likely to convert due to strong add-to-cart activity"
        if high_engagement:
            return "Likely to convert due to high session engagement"
        if recent_activity:
            return "Likely to convert due to recent user activity"
        return "Likely to convert based on overall behavioral patterns"

    else:
        if low_activity:
            return "Unlikely to convert due to low engagement"
        if no_cart:
            return "Unlikely to convert due to lack of purchase intent signals"
        return "Unlikely to convert based on weak behavioral signals"


@app.post("/predict")
def predict_conversion(user_features: UserFeaturesInput) -> dict:
    try:
        # Convert input → dataframe
        features_dataframe = pd.DataFrame([user_features.model_dump()])
        features_dataframe = features_dataframe[feature_columns]

        # Predict probability
        conversion_probability = float(
            model.predict_proba(features_dataframe)[0][1]
        )

        # Optional cap to avoid exact 1.0 display
        conversion_probability = min(conversion_probability, 0.999)

        # Apply threshold
        prediction = int(conversion_probability >= decision_threshold)

        # Generate explanation aligned with prediction
        reason = build_prediction_reason(user_features, prediction)

        return {
            "probability": round(conversion_probability, 4),
            "prediction": prediction,
            "label": "convert" if prediction == 1 else "not_convert",
            "reason": reason,
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error