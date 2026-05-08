from pathlib import Path

from src.data_preprocessing import preprocess_data
from src.train_model import train_failure_model
from src.optimize_maintenance import run_maintenance_optimization
from src.utils import print_section, ensure_directory


def main():
    """
    Run the full predictive maintenance pipeline:
    1. Load and preprocess raw data
    2. Train failure-risk model
    3. Save predictions and feature importance
    4. Run maintenance optimization
    5. Save final recommendations
    """

    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------
    raw_data_path = "data/raw/chemical_process_timeseries.csv"

    processed_data_path = "data/processed/ml_ready_data.csv"

    predictions_output_path = "outputs/results/failure_risk_predictions.csv"
    feature_importance_output_path = "outputs/results/rf_feature_importance.csv"
    final_recommendations_output_path = (
        "outputs/results/final_maintenance_recommendations.csv"
    )

    # ------------------------------------------------------------------
    # Make sure output folders exist
    # ------------------------------------------------------------------
    ensure_directory("data/processed")
    ensure_directory("outputs/results")
    ensure_directory("outputs/figures")

    # ------------------------------------------------------------------
    # Step 1: Preprocess data
    # ------------------------------------------------------------------
    print_section("Step 1: Data preprocessing")

    if not Path(raw_data_path).exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {raw_data_path}\n"
            "Please place the dataset file inside data/raw/."
        )

    df = preprocess_data(
        raw_file_path=raw_data_path,
        output_file_path=processed_data_path,
    )

    print(f"Processed data shape: {df.shape}")
    print(f"Processed data saved to: {processed_data_path}")

    # ------------------------------------------------------------------
    # Step 2: Train failure-risk model
    # ------------------------------------------------------------------
    print_section("Step 2: Failure-risk model training")

    (
        model,
        metrics,
        predictions,
        feature_importance,
        model_comparison,
        threshold_results,
    ) = train_failure_model(
        df=df,
        threshold=0.30,
        predictions_output_path=predictions_output_path,
        feature_importance_output_path=feature_importance_output_path,
        model_comparison_output_path="outputs/results/model_comparison.csv",
        threshold_analysis_output_path="outputs/results/threshold_analysis.csv",
    )

    print("Model training completed.")

    print("\nModel metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nClassification report:")
    print(metrics["classification_report"])

    print("\nModel comparison:")
    print(model_comparison)

    print("\nThreshold analysis:")
    print(threshold_results)

    print(f"Predictions saved to: {predictions_output_path}")
    print(f"Feature importance saved to: {feature_importance_output_path}")
    print("Model comparison saved to: outputs/results/model_comparison.csv")
    print("Threshold analysis saved to: outputs/results/threshold_analysis.csv")
    # ------------------------------------------------------------------
    # Step 3: Maintenance optimization
    # ------------------------------------------------------------------
    print_section("Step 3: Maintenance optimization")

    final_recommendations, optimization_summary = run_maintenance_optimization(
        predictions=predictions,
        output_path=final_recommendations_output_path,
    )

    print("Optimization completed.")
    print(f"Status: {optimization_summary['status']}")
    print(f"Objective value: {optimization_summary['objective_value']:.4f}")
    print(
        "Selected reactors:",
        optimization_summary["selected_reactors"],
    )

    print(f"Final recommendations saved to: {final_recommendations_output_path}")

    # ------------------------------------------------------------------
    # Step 4: Show final recommendation table
    # ------------------------------------------------------------------
    print_section("Final maintenance recommendations")

    display_cols = [
        "reactor_id",
        "avg_predicted_failure_risk",
        "avg_efficiency_loss",
        "min_time_to_fault",
        "maintenance_benefit",
        "maintenance_hours",
        "downtime_hours",
        "maintenance_cost",
        "selected_for_maintenance",
    ]

    available_cols = [
        col for col in display_cols
        if col in final_recommendations.columns
    ]

    print(final_recommendations[available_cols])

    print_section("Pipeline completed successfully")


if __name__ == "__main__":
    main()



