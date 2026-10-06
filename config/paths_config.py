from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
DATA_PATH = RAW_DATA_DIR / "healthcare-dataset-stroke-data.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
MLFLOW_DB = ROOT_DIR / "experiments" / "mlflow.db"

for directory in (PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
