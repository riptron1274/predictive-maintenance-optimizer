import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from pulp import (
    LpProblem,
    LpVariable,
    LpMaximize,
    lpSum,
    LpBinary,
    LpStatus,
    value,
)


def create_reactor_decision_input(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate time-step failure predictions into reactor-level decision inputs.

    Each row in the output represents one reactor.
    """

    pred = predictions.copy()

    # Treat zero or negative time-to-fault values as invalid for urgency calculation.
    pred["valid_time_to_fault"] = pred["time_to_fault_min"].where(
        pred["time_to_fault_min"] > 0
    )

    reactor_input = pred.groupby("reactor_id").agg(
        avg_predicted_failure_risk=("predicted_failure_risk", "mean"),
        max_predicted_failure_risk=("predicted_failure_risk", "max"),
        predicted_failure_count=("predicted_failure_label", "sum"),
        actual_failure_rate=("failure_risk", "mean"),
        avg_efficiency_loss=("efficiency_loss_pct", "mean"),
        min_time_to_fault=("valid_time_to_fault", "min"),
        avg_time_to_fault=("valid_time_to_fault", "mean"),
    ).reset_index()

    return reactor_input


def add_maintenance_parameters(
    reactor_input: pd.DataFrame,
    maintenance_hours: int = 3,
    downtime_hours: int = 2,
    maintenance_cost: int = 1000,
) -> pd.DataFrame:
    """
    Add assumed maintenance planning parameters.

    These are assumed values because the dataset does not include real maintenance
    cost, technician-hour, or downtime requirement data.
    """

    reactor_input = reactor_input.copy()

    reactor_input["maintenance_hours"] = maintenance_hours
    reactor_input["downtime_hours"] = downtime_hours
    reactor_input["maintenance_cost"] = maintenance_cost

    return reactor_input


def calculate_maintenance_benefit(
    reactor_input: pd.DataFrame,
    risk_weight: float = 0.50,
    efficiency_weight: float = 0.30,
    urgency_weight: float = 0.20,
) -> pd.DataFrame:
    """
    Calculate a normalized maintenance benefit score.

    The score combines:
    - predicted failure risk
    - expected efficiency loss
    - urgency based on minimum valid time-to-fault

    Higher score means higher maintenance priority.
    """

    reactor_input = reactor_input.copy()

    # Smaller time-to-fault should mean higher urgency.
    reactor_input["urgency_score"] = 1 / (
        reactor_input["min_time_to_fault"] + 1
    )

    # If min_time_to_fault is missing, urgency is treated as zero.
    reactor_input["urgency_score"] = reactor_input["urgency_score"].fillna(0)

    score_cols = [
        "avg_predicted_failure_risk",
        "avg_efficiency_loss",
        "urgency_score",
    ]

    scaler = MinMaxScaler()
    scaled_scores = scaler.fit_transform(reactor_input[score_cols])

    reactor_input["risk_score_norm"] = scaled_scores[:, 0]
    reactor_input["efficiency_loss_norm"] = scaled_scores[:, 1]
    reactor_input["urgency_score_norm"] = scaled_scores[:, 2]

    reactor_input["maintenance_benefit"] = (
        risk_weight * reactor_input["risk_score_norm"]
        + efficiency_weight * reactor_input["efficiency_loss_norm"]
        + urgency_weight * reactor_input["urgency_score_norm"]
    )

    reactor_input = reactor_input.sort_values(
        "maintenance_benefit",
        ascending=False,
    )

    return reactor_input


def solve_maintenance_optimization(
    reactor_input: pd.DataFrame,
    available_technician_hours: int = 12,
    allowed_downtime_hours: int = 6,
    available_budget: int = 4000,
) -> tuple[pd.DataFrame, dict]:
    """
    Solve the binary maintenance optimization model.

    Decision variable:
        x[i] = 1 if reactor i is selected for maintenance
        x[i] = 0 otherwise

    Objective:
        Maximize total maintenance benefit.

    Constraints:
        - technician hours
        - downtime hours
        - maintenance budget
    """

    reactor_input = reactor_input.copy()

    reactors = reactor_input["reactor_id"].tolist()

    model = LpProblem(
        "Predictive_Maintenance_Optimization",
        LpMaximize,
    )

    x = {
        r: LpVariable(f"maintain_{r}", cat=LpBinary)
        for r in reactors
    }

    # Objective: maximize total maintenance benefit.
    model += lpSum(
        reactor_input.loc[
            reactor_input["reactor_id"] == r,
            "maintenance_benefit",
        ].values[0] * x[r]
        for r in reactors
    )

    # Technician-hour constraint.
    model += lpSum(
        reactor_input.loc[
            reactor_input["reactor_id"] == r,
            "maintenance_hours",
        ].values[0] * x[r]
        for r in reactors
    ) <= available_technician_hours

    # Downtime constraint.
    model += lpSum(
        reactor_input.loc[
            reactor_input["reactor_id"] == r,
            "downtime_hours",
        ].values[0] * x[r]
        for r in reactors
    ) <= allowed_downtime_hours

    # Budget constraint.
    model += lpSum(
        reactor_input.loc[
            reactor_input["reactor_id"] == r,
            "maintenance_cost",
        ].values[0] * x[r]
        for r in reactors
    ) <= available_budget

    model.solve()

    results = reactor_input.copy()

    results["selected_for_maintenance"] = results["reactor_id"].apply(
        lambda r: int(x[r].value())
    )

    results = results.sort_values(
        ["selected_for_maintenance", "maintenance_benefit"],
        ascending=[False, False],
    )

    summary = {
        "status": LpStatus[model.status],
        "objective_value": value(model.objective),
        "available_technician_hours": available_technician_hours,
        "allowed_downtime_hours": allowed_downtime_hours,
        "available_budget": available_budget,
        "selected_reactors": results.loc[
            results["selected_for_maintenance"] == 1,
            "reactor_id",
        ].tolist(),
        "total_maintenance_hours_used": results.loc[
            results["selected_for_maintenance"] == 1,
            "maintenance_hours",
        ].sum(),
        "total_downtime_hours_used": results.loc[
            results["selected_for_maintenance"] == 1,
            "downtime_hours",
        ].sum(),
        "total_budget_used": results.loc[
            results["selected_for_maintenance"] == 1,
            "maintenance_cost",
        ].sum(),
    }

    return results, summary


def run_maintenance_optimization(
    predictions: pd.DataFrame,
    output_path: str | None = None,
    available_technician_hours: int = 12,
    allowed_downtime_hours: int = 6,
    available_budget: int = 4000,
) -> tuple[pd.DataFrame, dict]:
    """
    Full maintenance optimization pipeline.

    Steps:
    1. Aggregate ML predictions to reactor level
    2. Add maintenance parameters
    3. Calculate maintenance benefit score
    4. Solve optimization model
    5. Save results if output path is provided
    """

    reactor_input = create_reactor_decision_input(predictions)

    reactor_input = add_maintenance_parameters(reactor_input)

    reactor_input = calculate_maintenance_benefit(reactor_input)

    results, summary = solve_maintenance_optimization(
        reactor_input=reactor_input,
        available_technician_hours=available_technician_hours,
        allowed_downtime_hours=allowed_downtime_hours,
        available_budget=available_budget,
    )

    if output_path is not None:
        results.to_csv(output_path, index=False)

    return results, summary