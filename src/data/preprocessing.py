from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def build_preprocessor(categorical_encoding="ordinal", numeric_features=None,
                       categorical_features=None):
    """Build fold-local preprocessing with ordinal or one-hot categorical encoding."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    if categorical_encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder_name = "ordinal"
    elif categorical_encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        encoder_name = "onehot"
    else:
        raise ValueError("categorical_encoding deve ser 'ordinal' ou 'onehot'")
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            (encoder_name, encoder)])
    numeric_columns = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_columns = (CATEGORICAL_FEATURES if categorical_features is None
                           else categorical_features)
    return ColumnTransformer([("num", numeric, numeric_columns),
                              ("cat", categorical, categorical_columns)])
