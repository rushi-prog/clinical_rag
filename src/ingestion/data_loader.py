import pandas as pd
from pathlib import Path

# Project root is 2 levels up from this file (src/ingestion/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_trials(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    csv_file = PROJECT_ROOT / "data" / "clinical_trials.csv"
    df = load_trials(csv_file)
    print(df.head())