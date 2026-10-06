import pandas as pd
from config.paths_config import DATA_PATH


def load_dataset(path=DATA_PATH):
    """Load raw CSV, interpreting common missing-value tokens as nulls."""
    frame = pd.read_csv(path, na_values=["N/A", "NA", ""])
    if "age" in frame.columns:
        frame["age"] = frame["age"].astype(int)
    return frame
