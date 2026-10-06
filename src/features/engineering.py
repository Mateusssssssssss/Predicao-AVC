import pandas as pd

from config.training_config import ID_COLUMN, TARGET_COLUMN

LOW_CORRELATION_THRESHOLD = 0.10


def target_correlations(features, target):
    """Absolute target correlations, mapping one-hot indicators to source columns."""
    frame = features.copy()
    categorical = frame.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    encoded = pd.get_dummies(frame, columns=categorical, dtype=int)
    correlations = encoded.apply(lambda column: column.corr(target)).abs()
    source_columns = {}
    for column in frame.columns:
        if column in categorical:
            encoded_names = [name for name in encoded.columns if name.startswith(f"{column}_")]
            source_columns[column] = correlations[encoded_names].max() if encoded_names else 0.0
        else:
            source_columns[column] = correlations.get(column, 0.0)
    return pd.Series(source_columns, dtype=float).sort_values(ascending=False)


def split_features_target(frame):
    """Remove ID (chave, não sinal clínico) e separa X/y."""
    X = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = frame[TARGET_COLUMN].astype(int)
    return X, y
