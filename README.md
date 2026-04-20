# User Conversion Prediction System

## Problem Statement
This project predicts whether a user will convert (purchase) from clickstream behavior.  
The business goal is to identify high-intent users early so product or marketing teams can prioritize interventions.

## Dataset Description
- Source format: Retailrocket event stream (`data/events.csv`)
- Key columns: `timestamp`, `visitorid`, `event`, `itemid`, `transactionid`
- Supported event types in preprocessing: `view`, `addtocart`, `transaction`
- Raw logs are converted into user-level features for supervised binary classification

## Project Structure

```text
project/
│── data/
│   └── events.csv
│── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│── models/
│── app.py
│── requirements.txt
│── README.md
```

## Feature Engineering
The pipeline sessionizes events using a 30-minute inactivity window and creates user-level behavioral features:

- `total_events`
- `total_views`
- `total_addtocart`
- `add_to_view_ratio`
- `activity_level`
- `total_sessions`
- `click_rate`
- `session_duration`
- `pages_per_session`
- `time_since_last_activity`
- `active_days`
- `cart_to_view_rate`
- `avg_events_per_session`

## Data Leakage Fix
`total_transactions` is used only to define the target:

- `converted = 1` if user has at least one transaction, else `0`

No transaction-derived signal is passed to model inputs. This prevents target leakage, which can inflate offline scores and fail in production.

## Model Training and Comparison
Two models are trained and compared:

1. **Calibrated Logistic Regression**
   - Base logistic model wrapped with `CalibratedClassifierCV(method="sigmoid")`
   - Produces better-calibrated probabilities for downstream decisioning

2. **Random Forest**
   - Non-linear baseline for comparison

Both models are evaluated with:
- Precision
- Recall
- F1-score
- Confusion Matrix

Best model is selected by F1-score and saved as `models/best_model.joblib`.

## Why F1-score
Conversion prediction is typically imbalanced (fewer converters than non-converters).  
F1-score balances precision and recall, making it a better single metric than accuracy for this use case.

## Probability Calibration and Thresholding
- Probabilities from Logistic Regression are calibrated using sigmoid calibration
- Decision threshold is configurable and defaults to `0.6` (stored in `models/metadata.json`)
- API uses the same saved threshold for consistent train/serve behavior

## Explainability
Training extracts top 5 logistic coefficients (by absolute magnitude) and stores them in metadata.  
API responses also include a lightweight human-readable reason based on behavior signals.

## How To Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train models and save artifacts
```bash
python src/train.py
```

### 3) Start API
```bash
uvicorn app:app --reload
```

### 4) Test API
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"total_events\":10,\"total_views\":8,\"total_addtocart\":2,\"add_to_view_ratio\":0.25,\"activity_level\":10,\"total_sessions\":3,\"click_rate\":0.2,\"session_duration\":4.1,\"pages_per_session\":2.7,\"time_since_last_activity\":12.0,\"active_days\":2.0,\"cart_to_view_rate\":0.25,\"avg_events_per_session\":3.3}"
```

Response includes:
- `probability`
- `prediction`
- `label`
- `reason`

## Notes
- `src/features.py` defines the single source of truth for model feature columns via `get_model_feature_columns()`
- `src/train.py` defaults to `max_rows=300_000` for practical local runs; increase this for fuller training
