RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET_COLUMN = "stroke"
ID_COLUMN = "id"
THRESHOLD_OBJECTIVE = "f1"
THRESHOLD_BETA = 2.0
# Rank candidates by threshold-independent AUPRC to favor minority-class separation.
MODEL_SELECTION_METRIC = "cv_average_precision_mean"
THRESHOLD_DEFAULT = 0.5
MLFLOW_EXPERIMENT = "stroke-risk-classification"
