import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "reactor_id",
    "operating_regime",
    "fault_type",
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
    "efficiency_loss_pct",
    "time_to_fault_min",
]


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw chemical process time-series dataset.
    """
    df = pd.read_csv(file_path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Check whether all required columns exist in the dataset.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - validate required columns
    - convert timestamp
    - sort by reactor and time
    - create binary failure target
    - interpolate missing numeric values by reactor
    """
    df = df.copy()

    validate_columns(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["reactor_id", "timestamp"])

    df["failure_risk"] = (df["fault_type"] != 0).astype(int)

    numeric_cols = df.select_dtypes(include=["number"]).columns

    df[numeric_cols] = (
        df.groupby("reactor_id")[numeric_cols]
        .transform(lambda x: x.interpolate().ffill().bfill())
    )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features from timestamp.
    """
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    return df


def add_rolling_features(
    df: pd.DataFrame,
    rolling_cols: list[str] | None = None,
    window: int = 60,
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation features by reactor.

    A 60-minute rolling window is used because the dataset is sampled every minute.
    These features help capture gradual sensor changes before faults.
    """
    df = df.copy()
    df = df.sort_values(["reactor_id", "timestamp"])

    if rolling_cols is None:
        rolling_cols = [
            "reactor_temp",
            "reactor_pressure",
            "coolant_flow_rate",
            "vibration_rms",
            "motor_current",
            "power_consumption_kw",
        ]

    for col in rolling_cols:
        df[f"{col}_rolling_mean_{window}"] = (
            df.groupby("reactor_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=min_periods).mean())
        )

        df[f"{col}_rolling_std_{window}"] = (
            df.groupby("reactor_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=min_periods).std())
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns

    df[numeric_cols] = (
        df.groupby("reactor_id")[numeric_cols]
        .transform(lambda x: x.ffill().bfill())
    )

    return df


def save_processed_data(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Save the processed dataset to CSV.
    """
    df.to_csv(output_file_path, index=False)


def preprocess_data(
    raw_file_path: str,
    output_file_path: str | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Load raw data
    2. Clean data
    3. Add time features
    4. Add rolling features
    5. Optionally save processed data
    """
    df = load_data(raw_file_path)
    df = clean_data(df)
    df = add_time_features(df)
    df = add_rolling_features(df)

    if output_file_path is not None:
        save_processed_data(df, output_file_path)

    return df