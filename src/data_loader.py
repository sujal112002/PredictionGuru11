from pathlib import Path
from typing import Optional

import pandas as pd


def load_user_events(data_path: str = "data/events.csv", max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load raw user events from CSV."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    dataframe = pd.read_csv(path, nrows=max_rows)
    return dataframe


def clean_user_events(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw events with assumptions that keep the pipeline predictable.
    """
    dataframe = raw_dataframe.copy()

    required_columns = ["timestamp", "visitorid", "event", "itemid"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    dataframe = dataframe.dropna(subset=["timestamp", "visitorid", "event"])

    dataframe["timestamp"] = pd.to_numeric(dataframe["timestamp"], errors="coerce")
    dataframe["visitorid"] = pd.to_numeric(dataframe["visitorid"], errors="coerce")
    dataframe["itemid"] = pd.to_numeric(dataframe["itemid"], errors="coerce")

    dataframe = dataframe.dropna(subset=["timestamp", "visitorid"])
    dataframe["event"] = dataframe["event"].astype(str).str.lower().str.strip()
    dataframe = dataframe[dataframe["event"].isin(["view", "addtocart", "transaction"])]

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms", errors="coerce")
    dataframe = dataframe.dropna(subset=["timestamp"])
    dataframe["visitorid"] = dataframe["visitorid"].astype("int64")
    dataframe["itemid"] = dataframe["itemid"].fillna(-1).astype("int64")

    dataframe = dataframe.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    return dataframe