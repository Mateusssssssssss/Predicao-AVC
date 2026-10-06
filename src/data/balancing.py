from imblearn.over_sampling import RandomOverSampler, SMOTENC
from src.data.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from config.training_config import RANDOM_STATE


def build_sampler(name, numeric_features=None, categorical_features=None):
    if name in {"none", "none_onehot"}:
        return "passthrough"
    try:
        ratio = int(name.rsplit("_", 1)[1]) / 100
    except (IndexError, ValueError):
        raise ValueError(f"Sampler desconhecido: {name}") from None
    if not 0 < ratio <= 1:
        raise ValueError(f"Proporção inválida para sampler: {name}")
    if name.startswith("random_over_"):
        return RandomOverSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
    numeric_columns = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_columns = (CATEGORICAL_FEATURES if categorical_features is None
                           else categorical_features)
    category_indices = list(range(len(numeric_columns),
                                  len(numeric_columns) + len(categorical_columns)))
    if name.startswith("smotenc_"):
        return SMOTENC(categorical_features=category_indices,
                       sampling_strategy=ratio,
                       random_state=RANDOM_STATE, k_neighbors=3)
    raise ValueError(f"Sampler desconhecido: {name}")
