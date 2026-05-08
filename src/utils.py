from pathlib import Path
import pandas as pd


def ensure_directory(path: str) -> None:
    """
    Create a directory if it does not already exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to CSV and create the parent folder if needed.
    """
    parent_dir = Path(file_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=False)


def print_section(title: str) -> None:
    """
    Print a formatted section header in the terminal.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_file_exists(file_path: str) -> None:
    """
    Raise an error if a required file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")


def get_project_paths() -> dict:
    """
    Return common project paths used in the pipeline.
    """
    return {
        "raw_data": "data/raw/",
        "processed_data": "data/processed/",
        "figures": "outputs/figures/",
        "results": "outputs/results/",
        "models": "outputs/models/",
    }