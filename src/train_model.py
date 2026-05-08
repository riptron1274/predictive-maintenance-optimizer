"""
train_model.py

Machine learning utilities for the predictive maintenance project.

This module trains failure-risk prediction models using processed
chemical reactor sensor data. It includes:

- Feature selection
- Time-based train/test split
- Logistic Regression baseline
- Random Forest model
- Model evaluation
- Threshold analysis
- Feature importance
- Prediction table generation for optimization
"""

from __future__ import annotations

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select model feature columns while avoiding target leakage.

    Excluded leakage columns include:
    - fault_type
    - failure_risk
    - efficiency_loss_pct
    - time_to_fault_min
    """

    base_features = [
        "ambient_temp_effect",
        "reactor_temp",
        "reactor_pressure",
        "feed_flow_rate",
        "coolant_flow_rate",
        "agitator_speed_rpm",
        "reaction_rate",
        "conversion_rate",
        "selectivity",
        "yield_pct",
        "vibration_rms",
        "motor_current",
        "power_consumption_kw",
        "temp_setpoint",
        "pressure_setpoint",
        "hour",
        "dayofweek",
        "month",
    ]

    rolling_features = [
        col
        for col in df.columns
        if "rolling_mean_60" in col or "rolling_std_60" in col
    ]

    feature_cols = base_features + rolling_features

    available_features = [
        col
        for col in feature_cols
        if col in df.columns
    ]

    return available_features


def create_time_based_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "failure_risk",
    train_ratio: float = 0.8,
):
    """
    Create a time-based train/test split.

    This is more realistic for predictive maintenance than a random split
    because the model is trained on earlier observations and tested on later
    observations.
    """

    df_model = df.copy()
    df_model["timestamp"] = pd.to_datetime(df_model["timestamp"])
    df_model = df_model.sort_values("timestamp")

    split_index = int(len(df_model) * train_ratio)

    train_df = df_model.iloc[:split_index].copy()
    test_df = df_model.iloc[split_index:].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    return train_df, test_df, X_train, X_test, y_train, y_test


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train a Logistic Regression baseline model.

    Logistic Regression is used as an interpretable baseline.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train an improved Random Forest classifier.

    Random Forest is used because industrial sensor behavior may contain
    nonlinear relationships and interactions between process variables.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.30,
) -> dict:
    """
    Evaluate a classification model using a selected probability threshold.

    In predictive maintenance, the threshold matters because missing a failure
    may be more costly than creating a false alarm.
    """

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


def compare_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Compare multiple classification models using the same threshold.
    """

    rows = []

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        "f1",
        ascending=False,
    )

    return comparison


def threshold_analysis(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate precision, recall, and F1-score under different thresholds.
    """

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    y_proba = model.predict_proba(X_test)[:, 1]

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def create_prediction_table(
    test_df: pd.DataFrame,
    model,
    X_test: pd.DataFrame,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Create a prediction table for the optimization model.

    This table connects the machine-learning layer to the maintenance
    optimization layer.
    """

    prediction_cols = [
        "timestamp",
        "reactor_id",
        "operating_regime",
        "fault_type",
        "failure_risk",
        "efficiency_loss_pct",
        "time_to_fault_min",
    ]

    available_cols = [
        col
        for col in prediction_cols
        if col in test_df.columns
    ]

    predictions = test_df[available_cols].copy()

    predictions["predicted_failure_risk"] = model.predict_proba(X_test)[:, 1]

    predictions["predicted_failure_label"] = (
        predictions["predicted_failure_risk"] >= threshold
    ).astype(int)

    return predictions


def get_feature_importance(
    model: RandomForestClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Return Random Forest feature importance.
    """

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return feature_importance

def leave_one_reactor_out_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Perform leave-one-reactor-out validation.

    Each reactor is used once as the test reactor while the model is trained
    on all other reactors.
    """
    results = []

    reactors = df["reactor_id"].unique()

    for test_reactor in reactors:
        train_df = df[df["reactor_id"] != test_reactor].copy()
        test_df = df[df["reactor_id"] == test_reactor].copy()

        X_train = train_df[feature_cols]
        y_train = train_df["failure_risk"]

        X_test = test_df[feature_cols]
        y_test = test_df["failure_risk"]

        model = train_random_forest(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        results.append({
            "test_reactor": test_reactor,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "actual_failure_rate": y_test.mean(),
        })

    return pd.DataFrame(results)
def train_failure_model(
    df: pd.DataFrame,
    threshold: float = 0.30,
    predictions_output_path: str | None = None,
    feature_importance_output_path: str | None = None,
    model_comparison_output_path: str | None = None,
    threshold_analysis_output_path: str | None = None,
):
    """
    Full failure-risk modeling pipeline.

    Steps:
    1. Select features
    2. Create time-based train/test split
    3. Train Logistic Regression baseline
    4. Train Random Forest model
    5. Compare models
    6. Evaluate final model
    7. Generate prediction table
    8. Generate feature importance
    9. Generate threshold analysis

    Returns:
        final_model,
        metrics,
        predictions,
        feature_importance,
        model_comparison,
        threshold_results
    """

    feature_cols = get_feature_columns(df)

    (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_time_based_split(
        df=df,
        feature_cols=feature_cols,
    )

    log_reg_model = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
    )

    rf_model = train_random_forest(
        X_train=X_train,
        y_train=y_train,
    )

    models = {
        "Logistic Regression": log_reg_model,
        "Random Forest": rf_model,
    }

    model_comparison = compare_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
    )

    final_model = rf_model

    metrics = evaluate_model(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
    )

    predictions = create_prediction_table(
        test_df=test_df,
        model=final_model,
        X_test=X_test,
        threshold=threshold,
    )

    feature_importance = get_feature_importance(
        model=final_model,
        feature_cols=feature_cols,
    )

    threshold_results = threshold_analysis(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
    )

    if predictions_output_path is not None:
        predictions.to_csv(predictions_output_path, index=False)

    if feature_importance_output_path is not None:
        feature_importance.to_csv(feature_importance_output_path, index=False)

    if model_comparison_output_path is not None:
        model_comparison.to_csv(model_comparison_output_path, index=False)

    if threshold_analysis_output_path is not None:
        threshold_results.to_csv(threshold_analysis_output_path, index=False)

    return (
        final_model,
        metrics,
        predictions,
        feature_importance,
        model_comparison,
        threshold_results,
    )