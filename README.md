# User Conversion Prediction System

Predicts whether an e-commerce visitor will make a purchase, using only their clickstream behavior (views and add-to-carts). The goal is to flag high-intent users early so product or marketing teams can prioritize them.

**Highlights:** session-based feature engineering · explicit target-leakage prevention · calibrated probabilities · F1-driven model selection on imbalanced data · FastAPI serving with per-prediction explanations

---

## Results

Evaluated on a held-out test set of **31,940 users**, of whom **254 converted (0.8%)**. Both models use a fixed decision threshold of 0.6.

| Model | Precision | Recall | F1 |
| --- | --- | --- | --- |
| **Random Forest (selected)** | **0.256** | **0.866** | **0.395** |
| Calibrated Logistic Regression | 0.229 | 0.075 | 0.113 |

**What this means:** the random forest flags about 2.7% of users and captures 87% of actual buyers. Its precision is roughly 32× the 0.8% base rate, so targeting flagged users is far more efficient than targeting at random.

**Random Forest confusion matrix:**

|  | Predicted: no purchase | Predicted: purchase |
| --- | --- | --- |
| **Actual: no purchase** | 31,046 | 640 |
| **Actual: purchase** | 34 | 220 |

**Strongest logistic-regression signals:** `click_rate` (+), `avg_events_per_session` (+), `pages_per_session` (−), `total_addtocart` (+), `session_duration` (+)

**Why logistic regression's recall is low:** calibration makes its probabilities realistic, and with a 0.8% base rate very few users ever reach a 60% predicted probability, so at a 0.6 threshold it flags only 83 users. Tuning each model's threshold on a validation split (see Next Steps) is the natural fix.

## Leakage I Caught

The first version of this project scored **F1 = 0.99**, which was too good to be true. The logistic-regression coefficients gave it away: large positive weight on `total_events`, large negative weights on `total_views` and `total_addtocart`. Because `total_events` included transaction events, the model could compute `total_events − total_views − total_addtocart` = number of purchases, which is the label itself.

The fix:
- Model features are computed only from `view` and `addtocart` events
- For converters, only events **before their first purchase** are used, matching how the model would be used in practice
- `activity_level` (a duplicate of `total_events`) and `time_since_last_activity` were removed

After the fix, F1 dropped from 0.99 to a realistic 0.39.

## Problem Statement

Conversion prediction from clickstream data: given a user's browsing behavior, estimate the probability that they will purchase. Converters are a small minority, so this is an imbalanced binary classification problem.

## Dataset

- Source: [Retailrocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (`data/events.csv`, not committed due to size)
- Key columns: `timestamp`, `visitorid`, `event`, `itemid`, `transactionid`
- Event types used: `view`, `addtocart`, `transaction`
- Raw event logs are aggregated into one row per user for supervised learning

## Project Structure

```
├── data/
│   └── events.csv          # download from Kaggle
├── src/
│   ├── data_loader.py      # load and clean raw events
│   ├── features.py         # sessionization + user-level features
│   ├── train.py            # train, compare, select, save
│   └── evaluate.py         # metrics and confusion matrix
├── models/                 # best_model.joblib, metadata.json
├── app.py                  # FastAPI prediction service
├── ui.py                   # simple front end
└── requirements.txt
```

## Feature Engineering

Using only pre-purchase `view` and `addtocart` events, activity is grouped into sessions with a 30-minute inactivity window, then aggregated per user:

| Group | Features |
| --- | --- |
| Volume | `total_events`, `total_views`, `total_addtocart` |
| Intent ratios | `add_to_view_ratio`, `cart_to_view_rate`, `click_rate` |
| Session behavior | `total_sessions`, `session_duration`, `pages_per_session`, `avg_events_per_session` |
| Frequency | `active_days` |

`src/features.py` is the single source of truth for model input columns (`get_model_feature_columns()`), so training and serving always use the same features.

## Preventing Data Leakage

Transactions are used **only** to build the label (`converted = 1` if the user has at least one transaction). No transaction event contributes to any feature, including aggregate counts and session statistics, and converters' histories are cut at their first purchase. See *Leakage I Caught* above for how an earlier version got this wrong.

## Modeling

1. **Calibrated Logistic Regression**: wrapped in `CalibratedClassifierCV(method="sigmoid")` so predicted probabilities can be trusted for decisions
2. **Random Forest**: non-linear baseline

Both are evaluated with precision, recall, F1 and a confusion matrix. The model with the best F1 (currently the random forest) is saved to `models/best_model.joblib`.

**Why F1, not accuracy?** With few converters, a model that predicts "no purchase" for everyone gets high accuracy and zero value. F1 balances catching converters (recall) against flagging too many non-converters (precision).

**Thresholding:** the decision threshold is a fixed `0.6`, stored in `models/metadata.json` so the API uses exactly the threshold from training. It was not tuned on the test set.

## Explainability

Training stores the top 5 logistic-regression coefficients (by absolute value) in the metadata. Each API response also includes a short human-readable reason based on the user's behavior.

## How to Run

```bash
pip install -r requirements.txt
python src/train.py              # trains, compares, saves artifacts
uvicorn app:app --reload         # starts the API
```

Test the API:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"total_events":10,"total_views":8,"total_addtocart":2,"add_to_view_ratio":0.25,"total_sessions":3,"click_rate":0.2,"session_duration":4.1,"pages_per_session":2.7,"active_days":2.0,"cart_to_view_rate":0.25,"avg_events_per_session":3.3}'
```

The response includes `probability`, `prediction`, `label` and `reason`.

## Limitations and Next Steps

- Training uses `max_rows=300_000` by default for fast local runs; full-data results may differ
- A time-based split (train on earlier weeks, test on later weeks) would better simulate real deployment
- Tune each model's decision threshold on a separate validation split (especially for the calibrated logistic regression)
- Possible extensions: gradient boosting (LightGBM/XGBoost), SHAP explanations, precision-recall curve for threshold tuning
