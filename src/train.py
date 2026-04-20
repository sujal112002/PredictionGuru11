import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import clean_user_events, load_user_events
from evaluate import analyze_model_errors, evaluate_predictions, print_model_comparison
from features import create_user_features, get_model_feature_columns

DEFAULT_DECISION_THRESHOLD = 0.6


def build_models() -> dict[str, object]:
    """Create baseline and tree models."""
    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    calibrated_logistic = CalibratedClassifierCV(
        estimator=logistic_pipeline,
        method="sigmoid",
        cv=3,
    )

    random_forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return {"logistic_regression_calibrated": calibrated_logistic, "random_forest": random_forest}


def predict_with_threshold(probabilities: pd.Series, threshold: float) -> pd.Series:
    """Convert probabilities into labels with a business-tunable threshold."""
    return (probabilities >= threshold).astype(int)


def extract_logistic_feature_importance(
    X_train: pd.DataFrame, y_train: pd.Series, feature_columns: list[str]
) -> list[dict[str, float]]:
    """
    Fit an interpretable logistic model and return top coefficients.
    """
    explanation_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    explanation_pipeline.fit(X_train, y_train)

    coefficients = explanation_pipeline.named_steps["classifier"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {"feature": feature_columns, "coefficient": coefficients}
    )
    coefficient_frame["abs_coefficient"] = coefficient_frame["coefficient"].abs()
    top_features = coefficient_frame.sort_values(
        "abs_coefficient", ascending=False
    ).head(5)

    importance_records = [
        {
            "feature": row["feature"],
            "coefficient": float(row["coefficient"]),
            "abs_coefficient": float(row["abs_coefficient"]),
        }
        for _, row in top_features.iterrows()
    ]
    return importance_records


def train_and_evaluate(
    max_rows: int | None = 300_000, decision_threshold: float = DEFAULT_DECISION_THRESHOLD
) -> dict[str, object]:
    raw_events = load_user_events(max_rows=max_rows)
    clean_events = clean_user_events(raw_events)
    user_features = create_user_features(clean_events)

    # Leakage makes offline metrics look better than real-world performance, so we only use behavior signals.
    feature_columns = get_model_feature_columns()
    X = user_features[feature_columns]
    y = user_features["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    evaluations: list[dict[str, object]] = []
    error_analysis_by_model: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = predict_with_threshold(probabilities, decision_threshold)

        evaluations.append(evaluate_predictions(model_name, y_test, predictions))
        error_analysis_by_model[model_name] = analyze_model_errors(X_test, y_test, predictions, probabilities)
        trained_models[model_name] = model

    logistic_feature_importance = extract_logistic_feature_importance(
        X_train, y_train, feature_columns
    )

    print("\nTop 5 Logistic Feature Coefficients (absolute magnitude)")
    for item in logistic_feature_importance:
        print(
            f"- {item['feature']}: coefficient={item['coefficient']:.4f}, "
            f"|coef|={item['abs_coefficient']:.4f}"
        )

    print_model_comparison(evaluations)
    return {
        "trained_models": trained_models,
        "evaluations": evaluations,
        "error_analysis": error_analysis_by_model,
        "feature_columns": feature_columns,
        "decision_threshold": decision_threshold,
        "logistic_feature_importance": logistic_feature_importance,
    }


def save_artifacts(training_results: dict[str, object], output_dir: str = "models") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    evaluations = training_results["evaluations"]
    best_model_name = max(evaluations, key=lambda item: item["f1_score"])["model_name"]
    best_model = training_results["trained_models"][best_model_name]

    joblib.dump(best_model, output_path / "best_model.joblib")

    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": training_results["feature_columns"],
        "evaluations": evaluations,
        "error_analysis": training_results["error_analysis"],
        "decision_threshold": training_results["decision_threshold"],
        "logistic_feature_importance": training_results["logistic_feature_importance"],
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved best model: {best_model_name}")
    print(f"Artifacts directory: {output_path.resolve()}")


if __name__ == "__main__":
    results = train_and_evaluate(max_rows=300_000)
    save_artifacts(results)