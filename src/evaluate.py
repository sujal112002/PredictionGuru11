from typing import Dict

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, object]:
    """Compute core classification metrics in a single place."""
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "model_name": model_name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": matrix.tolist(),
    }


def print_model_comparison(evaluation_results: list[Dict[str, object]]) -> None:
    """Print side-by-side model metrics in a readable format."""
    print("\nModel Performance Comparison")
    print("-" * 70)
    for result in evaluation_results:
        tn, fp, fn, tp = result["confusion_matrix"][0][0], result["confusion_matrix"][0][1], result["confusion_matrix"][1][0], result["confusion_matrix"][1][1]
        print(
            f"{result['model_name']:<20} "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f} | "
            f"F1: {result['f1_score']:.4f}"
        )
        print(f"{'':<20} TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
        print("-" * 70)


def analyze_model_errors(
    features_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    probabilities: pd.Series,
) -> Dict[str, object]:
    """
    Summarize where the model fails.
    """
    analysis_dataframe = features_test.copy()
    analysis_dataframe["actual"] = y_true.values
    analysis_dataframe["predicted"] = y_pred
    analysis_dataframe["probability"] = probabilities

    false_positives = analysis_dataframe[(analysis_dataframe["actual"] == 0) & (analysis_dataframe["predicted"] == 1)]
    false_negatives = analysis_dataframe[(analysis_dataframe["actual"] == 1) & (analysis_dataframe["predicted"] == 0)]

    return {
        "false_positive_count": int(len(false_positives)),
        "false_negative_count": int(len(false_negatives)),
        "false_positive_avg_click_rate": float(false_positives["click_rate"].mean()) if not false_positives.empty else 0.0,
        "false_negative_avg_click_rate": float(false_negatives["click_rate"].mean()) if not false_negatives.empty else 0.0,
        "false_positive_avg_pages_per_session": float(false_positives["pages_per_session"].mean()) if not false_positives.empty else 0.0,
        "false_negative_avg_pages_per_session": float(false_negatives["pages_per_session"].mean()) if not false_negatives.empty else 0.0,
    }
